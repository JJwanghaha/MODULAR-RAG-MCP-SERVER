"""通过工厂与图文调用验证 Azure 上游请求协议，模拟 HTTP。"""

import json
from dataclasses import replace

import httpx
import pytest

from src.core.settings import VisionLLMSettings, load_settings
from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import ImageInput
from src.libs.llm.llm_factory import LLMFactory

pytestmark = pytest.mark.unit


def vision_settings():
    """使用虚构服务地址和部署，不读取真实环境凭据。"""
    return replace(load_settings(), vision_llm=VisionLLMSettings(
        enabled=False, provider="azure", model="vision-model", max_image_size=2048,
        azure_endpoint="https://vision.example", api_version="test-version",
        deployment_name="vision-deployment",
    ))


def test_azure_vision_sends_text_image_and_history():
    """历史在前，当前用户文字和 Base64 图片在同一条消息内。"""
    def server(request):
        assert request.url.path == "/openai/deployments/vision-deployment/chat/completions"
        assert request.url.params["api-version"] == "test-version"
        assert request.headers["api-key"] == "test-key"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        assert payload == {
            "messages": [
                {"role": "system", "content": "用中文"},
                {"role": "assistant", "content": "上一轮回答"},
                {"role": "user", "content": [
                    {"type": "text", "text": "描述架构图"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,Zml4dHVyZQ=="}},
                ]},
            ], "temperature": 0.2, "max_tokens": 80,
        }
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "模拟图片描述"}}],
            "model": "test-vision", "usage": {"total_tokens": 7},
        })

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
        response = llm.chat_with_image(
            "描述架构图", ImageInput(base64="Zml4dHVyZQ=="),
            messages=[Message("system", "用中文"), Message("assistant", "上一轮回答")],
            temperature=0.2, max_tokens=80,
        )
        assert response.content == "模拟图片描述"
        assert response.model == "test-vision"
        assert response.usage == {"total_tokens": 7}
        assert isinstance(response, ChatResponse)
        assert not client.is_closed


def test_azure_preprocessing_resizes_path_image_and_preserves_original(tmp_path):
    """实际图片处理经公开方法验证，保持比例，不覆盖源文件。"""
    import io
    from PIL import Image

    path = tmp_path / "large.png"
    Image.new("RGB", (400, 200), "red").save(path)
    llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key")
    image = ImageInput(path=path)
    processed = llm.preprocess_image(image, max_size=(100, 100))

    assert processed.path is None
    assert processed.mime_type == "image/png"
    with Image.open(io.BytesIO(processed.data)) as resized:
        assert resized.size == (100, 50)
    with Image.open(path) as original:
        assert original.size == (400, 200)
    assert llm.preprocess_image(processed, max_size=(100, 100)) is processed


def test_azure_error_reports_service_code_without_server_message():
    """保留 Azure 错误码便于定位，不回显原始服务响应。"""
    def server(request):
        return httpx.Response(404, json={"error": {
            "code": "DeploymentNotFound", "message": "private-response-detail",
        }})

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
        with pytest.raises(RuntimeError) as error:
            llm.chat_with_image("描述", ImageInput(base64="Zml4dHVyZQ=="))

    assert "HTTP 404" in str(error.value)
    assert "DeploymentNotFound" in str(error.value)
    assert "private-response-detail" not in str(error.value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_azure_rejects_invalid_image_size_override(value):
    """构造覆盖值与类型化配置有同样的实际意义。"""
    with pytest.raises(RuntimeError, match="max_image_size"):
        LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", max_image_size=value)


@pytest.mark.parametrize("status", [401, 429, 503])
def test_azure_http_failures_are_explicit_without_response_details(status):
    from src.libs.llm.azure_vision_llm import AzureVisionLLMError

    def server(request):
        return httpx.Response(status, json={"error": {"code": "TestError", "message": "private-detail"}})

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
        with pytest.raises(AzureVisionLLMError, match=f"HTTP {status}") as error:
            llm.chat_with_image("描述", ImageInput(base64="Zml4dHVyZQ=="))
    assert "private-detail" not in str(error.value)


@pytest.mark.parametrize("exception", [httpx.ReadTimeout, httpx.ConnectError])
def test_azure_transport_errors_are_sanitized(exception):
    from src.libs.llm.azure_vision_llm import AzureVisionLLMError

    def server(request):
        raise exception("private-request-detail", request=request)

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
        with pytest.raises(AzureVisionLLMError, match=exception.__name__) as error:
            llm.chat_with_image("描述", ImageInput(base64="Zml4dHVyZQ=="))
    assert "private-request-detail" not in str(error.value)


@pytest.mark.parametrize("body", [
    {}, {"choices": []}, {"choices": [{"message": {"content": None}}]},
    {"choices": [{"message": {"content": " "}}]}, [],
])
def test_azure_rejects_invalid_text_response(body):
    from src.libs.llm.azure_vision_llm import AzureVisionLLMError

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
        with pytest.raises(AzureVisionLLMError, match="Invalid text response"):
            llm.chat_with_image("描述", ImageInput(base64="Zml4dHVyZQ=="))


def test_azure_rejects_non_json_response():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not-json"))) as client:
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
        with pytest.raises(RuntimeError, match="Invalid JSON response"):
            llm.chat_with_image("描述", ImageInput(base64="Zml4dHVyZQ=="))


@pytest.mark.parametrize("carrier", ["path", "data", "base64"])
def test_azure_image_carriers_reach_http_with_same_content(tmp_path, carrier):
    import base64
    import io
    from PIL import Image

    output = io.BytesIO()
    with Image.new("RGB", (40, 20), "green") as image:
        image.save(output, format="PNG")
    data = output.getvalue()
    path = tmp_path / "small.png"
    path.write_bytes(data)
    inputs = {"path": path, "data": data, "base64": base64.b64encode(data).decode("ascii")}

    def server(request):
        url = json.loads(request.content)["messages"][-1]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == data
        assert request.extensions["timeout"]["read"] == 60.0
        return httpx.Response(200, json={"choices": [{"message": {"content": "描述"}}]})

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
        assert llm.chat_with_image("描述", ImageInput(**{carrier: inputs[carrier]})).content == "描述"


def test_azure_base64_skips_preprocessing_and_mime_is_only_a_declaration():
    image = ImageInput(base64="Zml4dHVyZQ==", mime_type="image/jpeg")
    llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key")

    assert llm.preprocess_image(image, max_size=(1, 1)) is image


@pytest.mark.parametrize("image", [ImageInput(path="not-created.png"), ImageInput(data=b"invalid-image")])
def test_azure_image_failure_happens_before_request(image):
    def no_request(request):
        raise AssertionError("invalid image must not request server")

    with httpx.Client(transport=httpx.MockTransport(no_request)) as client:
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
        with pytest.raises(RuntimeError, match="Image preprocessing failed"):
            llm.chat_with_image("描述", image)


@pytest.mark.parametrize("text,image,history", [
    (" ", ImageInput(base64="Zml4dHVyZQ=="), None),
    ("描述", "not-an-ImageInput", None),
    ("描述", ImageInput(base64="Zml4dHVyZQ=="), [Message("tool", "不支持")]),
])
def test_azure_invalid_input_does_not_request_model(text, image, history):
    def no_request(request):
        raise AssertionError("invalid input must not request server")

    with httpx.Client(transport=httpx.MockTransport(no_request)) as client:
        llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", client=client)
        with pytest.raises(ValueError):
            llm.chat_with_image(text, image, messages=history)


def test_azure_does_not_assume_old_default_api_version(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    settings = vision_settings()
    settings = replace(settings, vision_llm=replace(settings.vision_llm, api_version=None))
    with pytest.raises(RuntimeError, match="Missing endpoint or api_version"):
        LLMFactory.create_vision_llm(settings, api_key="test-key")


def test_azure_reads_named_environment_credentials_without_contacting_service(monkeypatch):
    for name, value in {
        "AZURE_OPENAI_API_KEY": "test-key", "AZURE_OPENAI_ENDPOINT": "https://env.example",
        "AZURE_OPENAI_API_VERSION": "env-version",
    }.items():
        monkeypatch.setenv(name, value)
    settings = vision_settings()
    settings = replace(settings, vision_llm=replace(settings.vision_llm, azure_endpoint=None, api_version=None))

    def server(request):
        assert request.url.host == "env.example"
        assert request.url.params["api-version"] == "env-version"
        assert request.headers["api-key"] == "test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": "描述"}}]})

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        llm = LLMFactory.create_vision_llm(settings, client=client)
        assert llm.chat_with_image("描述", ImageInput(base64="Zml4dHVyZQ==")).content == "描述"


@pytest.mark.parametrize("provider,key_env", [("azure", "AZURE_OPENAI_API_KEY"), ("gemini", "GEMINI_API_KEY")])
def test_selected_vision_provider_requires_only_its_credential(monkeypatch, provider, key_env):
    monkeypatch.delenv(key_env, raising=False)
    settings = vision_settings()
    settings = replace(settings, vision_llm=replace(settings.vision_llm, provider=provider))
    with pytest.raises(RuntimeError, match=key_env):
        LLMFactory.create_vision_llm(settings)


@pytest.mark.parametrize("value", [0, float("nan"), True])
def test_azure_rejects_invalid_timeout_override(value):
    with pytest.raises(RuntimeError, match="timeout"):
        LLMFactory.create_vision_llm(vision_settings(), api_key="test-key", timeout=value)


@pytest.mark.parametrize("provider", ["azure", "gemini"])
@pytest.mark.parametrize("format,mime", [("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_vision_preprocessing_preserves_declared_format_and_mime(provider, format, mime):
    import io
    from PIL import Image

    output = io.BytesIO()
    with Image.new("RGB", (400, 200), "red") as image:
        image.save(output, format=format)
    settings = vision_settings()
    settings = replace(settings, vision_llm=replace(settings.vision_llm, provider=provider))
    llm = LLMFactory.create_vision_llm(settings, api_key="test-key")
    processed = llm.preprocess_image(ImageInput(data=output.getvalue(), mime_type=mime), max_size=(100, 100))

    assert processed.mime_type == mime
    with Image.open(io.BytesIO(processed.data)) as image:
        assert image.size == (100, 50)
        assert image.format == format


def test_azure_closes_its_own_http_client(monkeypatch):
    """自有 client 在单次请求后关闭；通过外部 HTTP seam 观察生命周期。"""
    original_client = httpx.Client
    clients = []

    def server(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "描述"}}]})

    def create_client(**kwargs):
        assert kwargs["timeout"] == 60.0
        client = original_client(transport=httpx.MockTransport(server), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", create_client)
    llm = LLMFactory.create_vision_llm(vision_settings(), api_key="test-key")

    assert llm.chat_with_image("描述", ImageInput(base64="Zml4dHVyZQ==")).content == "描述"
    assert clients[0].is_closed
