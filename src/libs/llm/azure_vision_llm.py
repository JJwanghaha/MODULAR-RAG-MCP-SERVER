"""复现上游 Azure 图文消息协议，使用现有 httpx 请求 seam。"""

import base64
import os
import re
from math import isfinite

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm._vision_images import image_bytes, resize_image


class AzureVisionLLMError(RuntimeError):
    """Azure 请求、图片编码或响应失败，不回显图片或服务原文。"""


class AzureVisionLLM(BaseVisionLLM):
    """调用配置的 Azure 部署；注入的 HTTP client 由调用方关闭。"""

    def __init__(self, settings, api_key=None, endpoint=None, deployment_name=None,
                 api_version=None, max_image_size=None, client=None, **kwargs):
        self.settings = settings
        vision = settings.vision_llm
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = (endpoint or getattr(vision, "azure_endpoint", None)
                         or getattr(vision, "base_url", None) or os.environ.get("AZURE_OPENAI_ENDPOINT")
                         or settings.llm.base_url)
        self.api_version = (api_version or getattr(vision, "api_version", None)
                            or os.environ.get("AZURE_OPENAI_API_VERSION") or settings.llm.api_version)
        self.deployment_name = (deployment_name or getattr(vision, "deployment_name", None)
                                or getattr(vision, "model", None) or settings.llm.model)
        self.max_image_size = max_image_size if max_image_size is not None else getattr(vision, "max_image_size", 2048)
        self.timeout = kwargs.get("timeout", settings.llm.timeout)
        self.client = client
        if not self.api_key:
            raise ValueError("[azure vision] Missing API key: set AZURE_OPENAI_API_KEY")
        if not self.endpoint or not self.api_version:
            raise ValueError("[azure vision] Missing endpoint or api_version")
        if not isinstance(self.max_image_size, int) or isinstance(self.max_image_size, bool) or self.max_image_size <= 0:
            raise ValueError("[azure vision] max_image_size must be a positive integer")
        if not isinstance(self.timeout, (int, float)) or isinstance(self.timeout, bool) or not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("[azure vision] timeout must be positive and finite")

    def preprocess_image(self, image, max_size=None):
        """处理路径/bytes 图片；保持上游 Base64 不缩放的限制。"""
        self.validate_image(image)
        try:
            return resize_image(image, max_size)
        except (OSError, ValueError, TypeError) as exc:
            raise AzureVisionLLMError(f"[azure vision] Image preprocessing failed: {type(exc).__name__}") from None

    def chat_with_image(self, text, image, messages=None, trace=None, **kwargs):
        """历史在前，当前文字与一张图片组合为同一条 user 消息。"""
        self.validate_text(text)
        self.validate_image(image)
        if messages:
            BaseLLM.validate_messages(self, messages)
        processed = self.preprocess_image(image, (self.max_image_size, self.max_image_size))
        try:
            encoded = processed.base64
            if encoded is None:
                data = image_bytes(processed)
                encoded = base64.b64encode(data).decode("ascii")
        except (OSError, TypeError, ValueError) as exc:
            raise AzureVisionLLMError(f"[azure vision] Image encoding failed: {type(exc).__name__}") from None
        api_messages = [{"role": m.role, "content": m.content} for m in messages or []]
        api_messages.append({"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{processed.mime_type};base64,{encoded}"}},
        ]})
        payload = {
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.settings.llm.temperature),
            "max_tokens": kwargs.get("max_tokens", self.settings.llm.max_tokens),
        }
        deployment = kwargs.get("deployment_name", self.deployment_name)
        url = f"{self.endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
        options = {"json": payload, "headers": {"api-key": self.api_key},
                   "params": {"api-version": self.api_version}}
        try:
            if self.client is not None:
                response = self.client.post(url, timeout=self.timeout, **options)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, **options)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            code = None
            try:
                code = exc.response.json().get("error", {}).get("code")
            except (ValueError, AttributeError, TypeError):
                pass
            suffix = f" code={code}" if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code) else ""
            raise AzureVisionLLMError(f"[azure vision] HTTP {exc.response.status_code}{suffix}") from None
        except httpx.RequestError as exc:
            raise AzureVisionLLMError(f"[azure vision] {type(exc).__name__}") from None
        except ValueError:
            raise AzureVisionLLMError("[azure vision] Invalid JSON response") from None
        try:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty text")
            return ChatResponse(content, data.get("model", deployment), data.get("usage"), data)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise AzureVisionLLMError("[azure vision] Invalid text response") from None
