"""上游 Ollama adapter 的新版批量 /api/embed 实现。"""

import os

import httpx

from src.libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbeddingError(RuntimeError):
    """本地编码服务失败或返回无效向量。"""


class OllamaEmbedding(BaseEmbedding):
    """不需要密钥；安装运行环境和下载模型由后续真实验收决定。"""

    def __init__(self, settings, base_url=None, client=None, **kwargs):
        self.model = settings.embedding.model
        self.dimensions = kwargs.get("dimensions", settings.embedding.dimensions)
        self.base_url = base_url or getattr(settings.embedding, "base_url", None) or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        self.timeout = kwargs.get("timeout", getattr(settings.embedding, "timeout", 60.0))
        self.client = client

    def embed(self, texts, trace=None, **kwargs):
        """批量输入，并拒绝服务自动截断超长文本。"""
        self.validate_texts(texts)
        payload = {"model": self.model, "input": texts, "truncate": False}
        url = f"{self.base_url.rstrip('/')}/api/embed"
        try:
            if self.client is not None:
                response = self.client.post(url, json=payload, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise OllamaEmbeddingError(f"[ollama] HTTP {exc.response.status_code}") from None
        except httpx.RequestError as exc:
            raise OllamaEmbeddingError(f"[ollama] {type(exc).__name__}") from None
        except ValueError:
            raise OllamaEmbeddingError("[ollama] Invalid JSON response") from None
        try:
            vectors = data["embeddings"]
            self.validate_vectors(vectors, len(texts))
            return vectors
        except (KeyError, TypeError, ValueError):
            raise OllamaEmbeddingError("[ollama] Invalid embedding output") from None

    def get_dimension(self):
        """配置维度须与所选模型真实输出一致。"""
        return self.dimensions
