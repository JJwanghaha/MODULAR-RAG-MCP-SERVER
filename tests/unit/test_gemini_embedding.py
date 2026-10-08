"""用真实 Google SDK 验证独立 Content 的批量向量请求。"""

import json
from dataclasses import replace

import httpx
import pytest
from google import genai
from google.genai import types

from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("task,expected", [
    ("RETRIEVAL_DOCUMENT", ["title: none | text: 甲", "title: none | text: 乙"]),
    ("RETRIEVAL_QUERY", ["task: search result | query: 甲", "task: search result | query: 乙"]),
])
def test_gemini_embeddings_keep_documents_separate_and_use_matching_task(task, expected):
    """两段文本必须返回两条向量；查询与文档采用配套检索格式。"""
    settings = load_settings()
    settings = replace(settings, embedding=replace(settings.embedding, provider="gemini", model="gemini-embedding-2", dimensions=2))

    def server(request):
        assert request.url.path == "/v1beta/models/gemini-embedding-2:batchEmbedContents"
        body = json.loads(request.content)
        assert len(body["requests"]) == 2
        assert [item["content"]["parts"][0]["text"] for item in body["requests"]] == expected
        assert all(item["outputDimensionality"] == 2 for item in body["requests"])
        assert all("taskType" not in item for item in body["requests"])
        return httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.0]}, {"values": [0.0, 1.0]}]})

    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        with genai.Client(api_key="test-key", vertexai=False, http_options=types.HttpOptions(httpx_client=http_client)) as client:
            embedding = EmbeddingFactory.create(settings, api_key="test-key", client=client)
            assert embedding.embed(["甲", "乙"], task_type=task) == [[1.0, 0.0], [0.0, 1.0]]
            assert embedding.get_dimension() == 2


def test_older_gemini_embedding_uses_task_type_and_normalizes_reduced_dimensions():
    """旧版模型使用 taskType，降维向量按官方要求归一化。"""
    settings = load_settings()
    settings = replace(settings, embedding=replace(settings.embedding, provider="gemini", model="gemini-embedding-001", dimensions=2))

    def server(request):
        body = json.loads(request.content)
        assert body["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
        assert body["requests"][0]["content"]["parts"] == [{"text": "问题"}]
        return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0]}]})

    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        with genai.Client(api_key="test-key", vertexai=False, http_options=types.HttpOptions(httpx_client=http_client)) as client:
            embedding = EmbeddingFactory.create(settings, api_key="test-key", client=client)
            assert embedding.embed(["问题"], task_type="RETRIEVAL_QUERY") == [[0.6, 0.8]]
