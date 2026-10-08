"""真实 SDK + 模拟 HTTP，验证云端向量的数量、顺序和维度。"""

import json
from dataclasses import replace

import httpx
import pytest
from openai import OpenAI, AzureOpenAI

from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_cloud_embedding_preserves_input_order_and_dimension(provider):
    """即使服务按索引乱序返回，调用者仍拿到与输入对应的向量。"""
    settings = load_settings()
    settings = replace(settings, embedding=replace(settings.embedding, provider=provider, model="text-embedding-3-small", dimensions=2))

    def server(request):
        body = json.loads(request.content)
        assert body["input"] == ["文档甲", "文档乙"]
        assert body["dimensions"] == 2
        assert body["encoding_format"] == "float"
        if provider == "azure":
            assert request.url.path == "/openai/deployments/embed-deployment/embeddings"
            assert request.url.params["api-version"] == "test-version"
            assert request.headers["api-key"] == "test-key"
        else:
            assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={
            "object": "list", "model": "text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
            ], "usage": {"prompt_tokens": 2, "total_tokens": 2},
        })

    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        options = {"api_key": "test-key", "http_client": http_client, "max_retries": 0}
        if provider == "azure":
            client = AzureOpenAI(**options, azure_endpoint="https://azure.example", azure_deployment="embed-deployment", api_version="test-version")
        else:
            client = OpenAI(**options)
        embedding = EmbeddingFactory.create(
            settings, api_key="test-key", client=client,
            endpoint="https://azure.example", api_version="test-version", deployment_name="embed-deployment",
        )
        assert embedding.embed(["文档甲", "文档乙"]) == [[1.0, 0.0], [0.0, 1.0]]
        assert embedding.get_dimension() == 2
