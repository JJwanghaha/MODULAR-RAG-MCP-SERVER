"""新增 Gemini 向量 adapter，兼容上游批量编码接口。"""

import os
from contextlib import ExitStack
from math import sqrt

from src.libs.embedding.base_embedding import BaseEmbedding


class GeminiEmbeddingError(RuntimeError):
    """Google 向量服务调用失败或输出不符合契约。"""


class GeminiEmbedding(BaseEmbedding):
    """每段文本独立 Content，避免 Embedding 2 聚合多个文档。"""

    def __init__(self, settings, api_key=None, client=None, **kwargs):
        self.model = settings.embedding.model
        self.dimensions = kwargs.get("dimensions", settings.embedding.dimensions)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("[gemini] Missing API key: set GEMINI_API_KEY")
        self.client = client
        self.timeout = kwargs.get("timeout", getattr(settings.embedding, "timeout", 60.0))
        self.base_url = kwargs.get("base_url", getattr(settings.embedding, "base_url", None))

    def embed(self, texts, trace=None, **kwargs):
        """默认编码文档；检索阶段显式传 task_type=RETRIEVAL_QUERY。"""
        self.validate_texts(texts)
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise GeminiEmbeddingError("[gemini] Missing SDK: install .[providers]") from None

        task = kwargs.get("task_type", "RETRIEVAL_DOCUMENT")
        config = {"output_dimensionality": self.dimensions}
        if self.model.startswith("gemini-embedding-2"):
            if task == "RETRIEVAL_DOCUMENT":
                texts = [f"title: none | text: {text}" for text in texts]
            elif task == "RETRIEVAL_QUERY":
                texts = [f"task: search result | query: {text}" for text in texts]
            else:
                raise ValueError("[gemini] Supported tasks: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY")
        else:
            config["task_type"] = task
        contents = [types.Content(parts=[types.Part(text=text)]) for text in texts]
        try:
            with ExitStack() as stack:
                client = self.client
                if client is None:
                    options = types.HttpOptions(
                        base_url=self.base_url, timeout=int(self.timeout * 1000),
                        retry_options=types.HttpRetryOptions(attempts=1),
                    )
                    client = stack.enter_context(genai.Client(api_key=self.api_key, vertexai=False, http_options=options))
                response = client.models.embed_content(model=self.model, contents=contents, config=types.EmbedContentConfig(**config))
        except Exception as exc:
            status = getattr(exc, "code", None)
            raise GeminiEmbeddingError(f"[gemini] {type(exc).__name__}" + (f" HTTP {status}" if status else "")) from None
        try:
            vectors = [item.values for item in response.embeddings]
            self.validate_vectors(vectors, len(texts))
            if self.model == "gemini-embedding-001" and self.dimensions != 3072:
                normalized = []
                for vector in vectors:
                    norm = sqrt(sum(value * value for value in vector))
                    if norm == 0:
                        raise ValueError("zero vector")
                    normalized.append([value / norm for value in vector])
                vectors = normalized
            return vectors
        except (AttributeError, TypeError, ValueError):
            raise GeminiEmbeddingError("[gemini] Invalid embedding output") from None

    def get_dimension(self):
        """输出维度由配置指定，并对真实返回值校验。"""
        return self.dimensions
