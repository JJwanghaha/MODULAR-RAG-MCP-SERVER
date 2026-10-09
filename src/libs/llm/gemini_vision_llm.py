"""Gemini 视觉扩展，原生图文 Content 与上游视觉 interface 对接。"""

import os
from contextlib import ExitStack
from math import isfinite

from src.libs.llm.base_llm import BaseLLM, ChatResponse
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm._vision_images import image_bytes, resize_image


class GeminiVisionLLMError(RuntimeError):
    """图片处理、Google 服务调用或文本响应失败。"""


class GeminiVisionLLM(BaseVisionLLM):
    """只发送单图和文字，不自动上传文件或切换模型供应商。"""

    def __init__(self, settings, api_key=None, client=None, max_image_size=None, **kwargs):
        self.settings = settings
        vision = settings.vision_llm
        self.model = getattr(vision, "model", None) or settings.llm.model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("[gemini vision] Missing API key: set GEMINI_API_KEY")
        self.client = client
        self.max_image_size = max_image_size if max_image_size is not None else getattr(vision, "max_image_size", 2048)
        self.timeout = kwargs.get("timeout", settings.llm.timeout)
        self.base_url = kwargs.get("base_url", getattr(vision, "base_url", None) or settings.llm.base_url)
        if not isinstance(self.max_image_size, int) or isinstance(self.max_image_size, bool) or self.max_image_size <= 0:
            raise ValueError("[gemini vision] max_image_size must be a positive integer")
        if not isinstance(self.timeout, (int, float)) or isinstance(self.timeout, bool) or not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("[gemini vision] timeout must be positive and finite")

    def preprocess_image(self, image, max_size=None):
        """复用同一缩放规则；Base64 载体沿用上游不缩放的限制。"""
        self.validate_image(image)
        try:
            return resize_image(image, max_size)
        except (OSError, ValueError, TypeError) as exc:
            raise GeminiVisionLLMError(f"[gemini vision] Image preprocessing failed: {type(exc).__name__}") from None

    def chat_with_image(self, text, image, messages=None, trace=None, **kwargs):
        """系统指令独立传递，历史 assistant 映射为 model，再追加图文输入。"""
        self.validate_text(text)
        self.validate_image(image)
        if messages:
            BaseLLM.validate_messages(self, messages)
        processed = self.preprocess_image(image, (self.max_image_size, self.max_image_size))
        try:
            data = image_bytes(processed)
        except (OSError, ValueError, TypeError) as exc:
            raise GeminiVisionLLMError(f"[gemini vision] Image encoding failed: {type(exc).__name__}") from None
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise GeminiVisionLLMError("[gemini vision] Missing SDK: install .[providers]") from None
        history = messages or []
        system = "\n\n".join(m.content for m in history if m.role == "system")
        contents = [types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.content)])
                    for m in history if m.role != "system"]
        contents.append(types.Content(role="user", parts=[
            types.Part(text=text), types.Part.from_bytes(data=data, mime_type=processed.mime_type),
        ]))
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=kwargs.get("temperature", self.settings.llm.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.settings.llm.max_tokens),
        )
        model = kwargs.get("model", self.model)
        try:
            with ExitStack() as stack:
                client = self.client
                if client is None:
                    options = types.HttpOptions(
                        base_url=self.base_url, timeout=int(self.timeout * 1000),
                        retry_options=types.HttpRetryOptions(attempts=1),
                    )
                    client = stack.enter_context(genai.Client(api_key=self.api_key, vertexai=False, http_options=options))
                response = client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            status = getattr(exc, "code", None)
            raise GeminiVisionLLMError(f"[gemini vision] {type(exc).__name__}" + (f" HTTP {status}" if status else "")) from None
        try:
            content = response.text
            if not isinstance(content, str) or not content.strip():
                raise ValueError("no text")
            usage = None
            if response.usage_metadata is not None:
                metadata = response.usage_metadata
                usage = {"prompt_tokens": metadata.prompt_token_count or 0,
                         "completion_tokens": metadata.candidates_token_count or 0,
                         "total_tokens": metadata.total_token_count or 0}
            return ChatResponse(content, response.model_version or model, usage, response.model_dump())
        except (AttributeError, TypeError, ValueError):
            raise GeminiVisionLLMError("[gemini vision] Invalid or blocked text response") from None
