"""所有后端通过公开入口验证失败行为，不连接真实服务。"""

from dataclasses import replace

import httpx
import pytest
from google import genai
from google.genai import types
from openai import OpenAI, AzureOpenAI

from src.core.settings import load_settings
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.embedding.embedding_factory import EmbeddingFactory

pytestmark = pytest.mark.unit
LLMS = ["openai", "azure", "deepseek", "ollama", "gemini"]
EMBEDDINGS = ["openai", "azure", "ollama", "gemini"]


@pytest.fixture
def adapter_factory():
    """仅替换外部 HTTP，真实 SDK 和 adapter 都运行。"""
    from contextlib import ExitStack

    with ExitStack() as stack:
        def create(kind, provider, server):
            settings = load_settings()
            options = replace(getattr(settings, kind), provider=provider, model="gemini-embedding-2" if kind == "embedding" and provider == "gemini" else "test-model")
            if kind == "embedding":
                options = replace(options, dimensions=2)
            settings = replace(settings, **{kind: options})
            http_client = stack.enter_context(httpx.Client(transport=httpx.MockTransport(server)))
            client = http_client
            if provider == "gemini":
                client = stack.enter_context(genai.Client(
                    api_key="secret-test-key", vertexai=False,
                    http_options=types.HttpOptions(httpx_client=http_client, retry_options=types.HttpRetryOptions(attempts=1)),
                ))
            elif kind == "embedding" and provider in {"openai", "azure"}:
                arguments = {"api_key": "secret-test-key", "http_client": http_client, "max_retries": 0}
                client = OpenAI(**arguments) if provider == "openai" else AzureOpenAI(
                    **arguments, azure_endpoint="https://azure.example", api_version="test-version",
                )
            factory = LLMFactory if kind == "llm" else EmbeddingFactory
            return factory.create(settings, api_key="secret-test-key", client=client, endpoint="https://azure.example", api_version="test-version")

        yield create


@pytest.mark.parametrize("provider", LLMS)
@pytest.mark.parametrize("failure", ["timeout", "connect", "http"])
def test_llm_transport_errors_are_readable_and_do_not_expose_key(adapter_factory, provider, failure):
    """超时、HTTP 错误只报告后端和错误类别，不回显服务错误正文。"""
    def server(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("secret-test-key", request=request)
        if failure == "connect":
            raise httpx.ConnectError("secret-test-key", request=request)
        return httpx.Response(401, json={"error": {"code": 401, "message": "secret-test-key", "status": "UNAUTHENTICATED"}})

    llm = adapter_factory("llm", provider, server)
    with pytest.raises(RuntimeError) as error:
        llm.chat([Message("user", "测试")])
    assert provider in str(error.value)
    assert "secret-test-key" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize("provider", EMBEDDINGS)
@pytest.mark.parametrize("failure", ["timeout", "connect", "http"])
def test_embedding_transport_errors_do_not_expose_key(adapter_factory, provider, failure):
    """向量接口沿用相同的错误信息安全边界。"""
    def server(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("secret-test-key", request=request)
        if failure == "connect":
            raise httpx.ConnectError("secret-test-key", request=request)
        return httpx.Response(401, json={"error": {"code": 401, "message": "secret-test-key", "status": "UNAUTHENTICATED"}})

    embedding = adapter_factory("embedding", provider, server)
    with pytest.raises(RuntimeError) as error:
        embedding.embed(["测试"])
    assert provider in str(error.value)
    assert "secret-test-key" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize("provider", LLMS)
def test_llm_rejects_empty_or_blocked_response(adapter_factory, provider):
    """空回答不能伪装成成功的生成结果。"""
    llm = adapter_factory("llm", provider, lambda request: httpx.Response(200, json={}))
    with pytest.raises(RuntimeError):
        llm.chat([Message("user", "测试")])


@pytest.mark.parametrize("provider", EMBEDDINGS)
@pytest.mark.parametrize("failure", ["count", "dimension"])
def test_embedding_rejects_mismatched_output(adapter_factory, provider, failure):
    """错误数量或维度必须在入库前被拒绝。"""
    def server(request):
        vector = [1.0] if failure == "dimension" else [1.0, 0.0]
        if provider == "gemini":
            return httpx.Response(200, json={"embeddings": [{"values": vector}]})
        if provider == "ollama":
            return httpx.Response(200, json={"embeddings": [vector]})
        return httpx.Response(200, json={
            "object": "list", "model": "test-model",
            "data": [{"object": "embedding", "index": 0, "embedding": vector}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    embedding = adapter_factory("embedding", provider, server)
    with pytest.raises(RuntimeError):
        embedding.embed(["甲", "乙"] if failure == "count" else ["甲"])


@pytest.mark.parametrize("kind,provider,key_env", [
    ("llm", "openai", "OPENAI_API_KEY"), ("llm", "deepseek", "DEEPSEEK_API_KEY"),
    ("llm", "azure", "AZURE_OPENAI_API_KEY"), ("llm", "gemini", "GEMINI_API_KEY"),
    ("embedding", "openai", "OPENAI_API_KEY"), ("embedding", "azure", "AZURE_OPENAI_API_KEY"),
    ("embedding", "gemini", "GEMINI_API_KEY"),
])
def test_only_selected_cloud_provider_requires_its_key(monkeypatch, kind, provider, key_env):
    """缺 Key 只阻止创建选中的后端，不影响导入和其他后端。"""
    monkeypatch.delenv(key_env, raising=False)
    settings = load_settings()
    settings = replace(settings, **{kind: replace(getattr(settings, kind), provider=provider)})
    factory = LLMFactory if kind == "llm" else EmbeddingFactory
    with pytest.raises(RuntimeError, match=key_env):
        factory.create(settings, endpoint="https://azure.example", api_version="test-version")


@pytest.mark.parametrize("provider", LLMS)
def test_invalid_messages_never_reach_network(adapter_factory, provider):
    """非法输入在 adapter 入口被拒绝。"""
    def forbidden(request):
        pytest.fail("非法输入不应发送请求")

    llm = adapter_factory("llm", provider, forbidden)
    with pytest.raises(ValueError):
        llm.chat([])


@pytest.mark.parametrize("provider", LLMS)
def test_non_string_message_content_has_readable_validation_error(adapter_factory, provider):
    """内容类型错误应报告输入问题，而不是泄露底层 AttributeError。"""
    llm = adapter_factory("llm", provider, lambda request: pytest.fail("不应发送请求"))
    with pytest.raises(ValueError, match="content"):
        llm.chat([Message("user", 42)])


@pytest.mark.parametrize("provider", EMBEDDINGS)
@pytest.mark.parametrize("texts", [[" "], [42], "not-a-list"])
def test_invalid_texts_never_reach_network(adapter_factory, provider, texts):
    """空文本不应消耗 API 额度。"""
    def forbidden(request):
        pytest.fail("非法输入不应发送请求")

    embedding = adapter_factory("embedding", provider, forbidden)
    with pytest.raises(ValueError):
        embedding.embed(texts)
