"""Azure 文本模型：保留上游部署路径、api-version 和 api-key 协议。"""

import os

import httpx

from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError


class AzureLLM(OpenAILLM):
    """复用消息转换，但单独处理 Azure 的 HTTP 请求。"""

    PROVIDER = "azure"
    KEY_ENV = "AZURE_OPENAI_API_KEY"

    def __init__(self, settings, endpoint=None, api_version=None, deployment_name=None, base_url=None, **kwargs):
        endpoint = endpoint or base_url or getattr(settings.llm, "base_url", None) or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or getattr(settings.llm, "api_version", None) or os.environ.get("AZURE_OPENAI_API_VERSION")
        self.deployment_name = deployment_name or getattr(settings.llm, "deployment_name", None) or settings.llm.model
        if not endpoint or not self.api_version:
            raise ValueError("[azure] Missing endpoint or api_version")
        super().__init__(settings, base_url=endpoint, **kwargs)

    def _call_api(self, payload):
        """Azure 使用部署路径及 api-key，不沿用 Bearer 认证。"""
        url = f"{self.base_url.rstrip('/')}/openai/deployments/{self.deployment_name}/chat/completions"
        params = {"api-version": self.api_version}
        headers = {"api-key": self.api_key}
        try:
            if self.client is not None:
                response = self.client.post(url, json=payload, headers=headers, params=params, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise OpenAILLMError(f"[azure] HTTP {exc.response.status_code}") from None
        except httpx.RequestError as exc:
            raise OpenAILLMError(f"[azure] {type(exc).__name__}") from None
        except ValueError:
            raise OpenAILLMError("[azure] Invalid JSON response") from None
