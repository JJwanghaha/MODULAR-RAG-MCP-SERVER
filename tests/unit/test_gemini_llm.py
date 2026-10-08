"""用真实 Google SDK 和模拟 HTTP 验证消息、配置与统一响应。"""

import json
from dataclasses import replace

import httpx
import pytest
from google import genai
from google.genai import types

from src.core.settings import load_settings
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory

pytestmark = pytest.mark.unit


def test_gemini_converts_system_and_history_without_server_session():
    """系统指令与多轮历史分别编码，回答及 usage 保持统一契约。"""
    settings = load_settings()
    settings = replace(settings, llm=replace(settings.llm, provider="gemini", model="test-gemini"))

    def server(request):
        assert request.url.path == "/v1beta/models/test-gemini:generateContent"
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["systemInstruction"]["parts"] == [{"text": "请用中文"}]
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "你好"}]},
            {"role": "model", "parts": [{"text": "你好呀"}]},
            {"role": "user", "parts": [{"text": "继续"}]},
        ]
        assert body["generationConfig"]["maxOutputTokens"] == 80
        assert body["generationConfig"]["temperature"] == 0.2
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "继续学习"}]}, "finishReason": "STOP"}],
            "modelVersion": "test-gemini",
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7},
        })

    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        with genai.Client(api_key="test-key", vertexai=False, http_options=types.HttpOptions(httpx_client=http_client)) as client:
            llm = LLMFactory.create(settings, api_key="test-key", client=client)
            result = llm.chat([
                Message("system", "请用中文"), Message("user", "你好"),
                Message("assistant", "你好呀"), Message("user", "继续"),
            ], max_tokens=80, temperature=0.2)
    assert result.content == "继续学习"
    assert result.model == "test-gemini"
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
