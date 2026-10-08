"""新增 Gemini adapter，保持上游 BaseLLM 接口不变。"""

import os
from contextlib import ExitStack

from src.libs.llm.base_llm import BaseLLM, ChatResponse


class GeminiLLMError(RuntimeError):
    """Gemini 调用失败或没有返回有效文本。"""


class GeminiLLM(BaseLLM):
    """用原生 SDK 的 generateContent 完成无状态文本调用。"""

    def __init__(self, settings, api_key=None, client=None, **kwargs):
        self.settings = settings
        self.model = settings.llm.model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("[gemini] Missing API key: set GEMINI_API_KEY")
        self.client = client
        self.timeout = kwargs.get("timeout", getattr(settings.llm, "timeout", 60.0))
        self.base_url = kwargs.get("base_url", getattr(settings.llm, "base_url", None))

    def chat(self, messages, trace=None, **kwargs):
        """系统指令独立传递，assistant 角色映射到 Gemini 的 model。"""
        self.validate_messages(messages)
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise GeminiLLMError("[gemini] Missing SDK: install .[providers]") from None

        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.content)])
            for m in messages if m.role != "system"
        ]
        if not contents:
            raise ValueError("[gemini] Messages must contain a user or assistant message")
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
            raise GeminiLLMError(f"[gemini] {type(exc).__name__}" + (f" HTTP {status}" if status else "")) from None
        content = response.text
        if not isinstance(content, str) or not content.strip():
            raise GeminiLLMError("[gemini] No text returned (blocked or invalid response)")
        usage = None
        if response.usage_metadata is not None:
            metadata = response.usage_metadata
            usage = {
                "prompt_tokens": metadata.prompt_token_count or 0,
                "completion_tokens": metadata.candidates_token_count or 0,
                "total_tokens": metadata.total_token_count or 0,
            }
        return ChatResponse(content, response.model_version or model, usage, response.model_dump())
