"""Azure Embedding 使用上游的 AzureOpenAI SDK 协议。"""

import os

from src.libs.embedding.openai_embedding import OpenAIEmbedding


class AzureEmbedding(OpenAIEmbedding):
    """复用批量输出校验，只替换 SDK 的连接配置。"""

    PROVIDER = "azure"
    KEY_ENV = "AZURE_OPENAI_API_KEY"

    def __init__(self, settings, endpoint=None, api_version=None, deployment_name=None, base_url=None, **kwargs):
        self.endpoint = endpoint or base_url or getattr(settings.embedding, "base_url", None) or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or getattr(settings.embedding, "api_version", None) or os.environ.get("AZURE_OPENAI_API_VERSION")
        self.deployment_name = deployment_name or getattr(settings.embedding, "deployment_name", None) or settings.embedding.model
        if not self.endpoint or not self.api_version:
            raise ValueError("[azure] Missing endpoint or api_version")
        super().__init__(settings, **kwargs)

    def _create_client(self):
        """不把 Azure 的部署名误当成 OpenAI 官方 URL。"""
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=self.api_key, azure_endpoint=self.endpoint,
            azure_deployment=self.deployment_name, api_version=self.api_version,
            timeout=self.timeout, max_retries=0,
        )
