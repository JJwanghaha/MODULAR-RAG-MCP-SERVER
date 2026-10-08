"""沿用上游 OpenAI SDK 路线，返回与输入顺序一致的向量。"""

import os
from contextlib import ExitStack

from src.libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbeddingError(RuntimeError):
    """向量服务请求失败或输出契约不匹配。"""


class OpenAIEmbedding(BaseEmbedding):
    """文本批量编码；SDK 在实际调用时才导入。"""

    PROVIDER = "openai"
    KEY_ENV = "OPENAI_API_KEY"

    def __init__(self, settings, api_key=None, base_url=None, client=None, **kwargs):
        self.settings = settings
        self.model = settings.embedding.model
        self.dimensions = kwargs.get("dimensions", settings.embedding.dimensions)
        self.api_key = api_key or os.environ.get(self.KEY_ENV)
        if not self.api_key:
            raise ValueError(f"[{self.PROVIDER}] Missing API key: set {self.KEY_ENV}")
        self.base_url = base_url or getattr(settings.embedding, "base_url", None) or "https://api.openai.com/v1"
        self.timeout = kwargs.get("timeout", getattr(settings.embedding, "timeout", 60.0))
        self.client = client

    def _create_client(self):
        """创建自有 SDK client；注入的 client 不由 adapter 关闭。"""
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)

    def embed(self, texts, trace=None, **kwargs):
        """浮点响应按 index 恢复顺序；旧模型不发送 dimensions 参数。"""
        self.validate_texts(texts)
        params = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self.model.startswith("text-embedding-3"):
            params["dimensions"] = self.dimensions
        try:
            with ExitStack() as stack:
                client = self.client if self.client is not None else stack.enter_context(self._create_client())
                response = client.embeddings.create(**params)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            raise OpenAIEmbeddingError(f"[{self.PROVIDER}] {type(exc).__name__}" + (f" HTTP {status}" if status else "")) from None
        try:
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(texts))):
                raise ValueError("invalid index")
            vectors = [item.embedding for item in items]
            self.validate_vectors(vectors, len(texts))
            return vectors
        except (AttributeError, TypeError, ValueError):
            raise OpenAIEmbeddingError(f"[{self.PROVIDER}] Invalid embedding count, index or dimension") from None

    def get_dimension(self):
        """维度来自配置，真实响应必须与它一致。"""
        return self.dimensions
