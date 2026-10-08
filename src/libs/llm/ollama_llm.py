"""沿用上游 Ollama /api/chat 路线，不安装或下载本地模型。"""

import os

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse


class OllamaLLMError(RuntimeError):
    """本地服务连接失败或响应无效。"""


class OllamaLLM(BaseLLM):
    """调用本地模型服务；不需要 API Key。"""

    def __init__(self, settings, base_url=None, client=None, **kwargs):
        self.settings = settings
        self.model = settings.llm.model
        self.base_url = base_url or getattr(settings.llm, "base_url", None) or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        self.timeout = kwargs.get("timeout", getattr(settings.llm, "timeout", 60.0))
        self.client = client

    def chat(self, messages, trace=None, **kwargs):
        """将 Ollama 的消息与 token 计数转换成统一响应。"""
        self.validate_messages(messages)
        model = kwargs.get("model", self.model)
        payload = {
            "model": model, "stream": False,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": kwargs.get("temperature", self.settings.llm.temperature),
                "num_predict": kwargs.get("max_tokens", self.settings.llm.max_tokens),
            },
        }
        url = f"{self.base_url.rstrip('/')}/api/chat"
        try:
            if self.client is not None:
                response = self.client.post(url, json=payload, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise OllamaLLMError(f"[ollama] HTTP {exc.response.status_code}") from None
        except httpx.RequestError as exc:
            raise OllamaLLMError(f"[ollama] {type(exc).__name__}") from None
        except ValueError:
            raise OllamaLLMError("[ollama] Invalid JSON response") from None
        try:
            content = data["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty text")
            prompt = data.get("prompt_eval_count", 0)
            completion = data.get("eval_count", 0)
            usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
            return ChatResponse(content, data.get("model", model), usage, data)
        except (KeyError, TypeError, AttributeError, ValueError):
            raise OllamaLLMError("[ollama] Invalid text response") from None
