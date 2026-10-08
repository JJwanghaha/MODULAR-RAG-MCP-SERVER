"""沿用上游的 HTTP 调用方式，封装 OpenAI-compatible 文本生成。"""

import os
from typing import Any

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OpenAILLMError(RuntimeError):
    """请求失败或服务响应不符合文本生成契约。"""


class OpenAILLM(BaseLLM):
    """将统一消息翻译成 Chat Completions 请求。"""

    PROVIDER = "openai"
    KEY_ENV = "OPENAI_API_KEY"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, settings, api_key=None, base_url=None, client=None, **kwargs):
        self.settings = settings
        self.model = settings.llm.model
        self.api_key = api_key or os.environ.get(self.KEY_ENV)
        if not self.api_key:
            raise ValueError(f"[{self.PROVIDER}] Missing API key: set {self.KEY_ENV}")
        self.base_url = base_url or getattr(settings.llm, "base_url", None) or self.DEFAULT_BASE_URL
        self.timeout = kwargs.get("timeout", getattr(settings.llm, "timeout", 60.0))
        self.client = client

    def chat(self, messages: list[Message], trace=None, **kwargs: Any) -> ChatResponse:
        """校验消息、发出请求，再统一响应；暂不支持流式及工具调用。"""
        self.validate_messages(messages)
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", self.settings.llm.temperature),
            "max_tokens": kwargs.get("max_tokens", self.settings.llm.max_tokens),
        }
        data = self._call_api(payload)
        try:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty text")
            return ChatResponse(content, data.get("model", payload["model"]), data.get("usage"), data)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise OpenAILLMError(f"[{self.PROVIDER}] Invalid text response") from None

    def _call_api(self, payload):
        """外部 HTTP 边界；传入的 client 由调用方管理生命周期。"""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            if self.client is not None:
                response = self.client.post(url, json=payload, headers=headers, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise OpenAILLMError(f"[{self.PROVIDER}] HTTP {exc.response.status_code}") from None
        except httpx.RequestError as exc:
            raise OpenAILLMError(f"[{self.PROVIDER}] {type(exc).__name__}") from None
        except ValueError:
            raise OpenAILLMError(f"[{self.PROVIDER}] Invalid JSON response") from None
