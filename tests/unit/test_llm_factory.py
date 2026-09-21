"""LLM interface 与 Factory 的公开行为测试。"""

from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.llm.llm_factory import LLMFactory


pytestmark = pytest.mark.unit


class FakeLLM(BaseLLM):
    """不访问网络、返回确定结果的测试 adapter。"""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def chat(self, messages, trace=None, **kwargs):
        """返回固定响应。"""
        self.validate_messages(messages)
        return ChatResponse(content="fake response", model="fake-model")


def _isolated_factory() -> type[LLMFactory]:
    """为每个测试创建独立 Registry。"""

    class IsolatedLLMFactory(LLMFactory):
        _PROVIDERS = {}

    return IsolatedLLMFactory


def _settings_with_provider(provider: str):
    """复用默认配置，只替换当前测试关注的 LLM provider。"""
    settings = load_settings()
    return replace(settings, llm=replace(settings.llm, provider=provider))


def test_fake_llm_returns_chat_response() -> None:
    """BaseLLM adapter 使用统一 Message 和 ChatResponse。"""
    llm = FakeLLM()

    response = llm.chat([Message(role="user", content="你好")])

    assert response == ChatResponse(content="fake response", model="fake-model")


def test_validate_messages_rejects_empty_list() -> None:
    """LLM 调用不能缺少消息。"""
    llm = FakeLLM()

    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        llm.validate_messages([])


def test_validate_messages_rejects_invalid_role() -> None:
    """消息角色必须属于统一枚举。"""
    llm = FakeLLM()
    messages = [Message(role="tool", content="结果")]

    with pytest.raises(ValueError, match="invalid role 'tool'"):
        llm.validate_messages(messages)


def test_validate_messages_rejects_blank_content() -> None:
    """纯空白内容不能进入 LLM adapter。"""
    llm = FakeLLM()
    messages = [Message(role="user", content="   ")]

    with pytest.raises(ValueError, match="empty content"):
        llm.validate_messages(messages)


def test_validate_messages_rejects_non_message_value() -> None:
    """字典等非 Message 对象不能穿过 LLM interface。"""
    llm = FakeLLM()

    with pytest.raises(ValueError, match="not a Message instance"):
        llm.validate_messages([{"role": "user", "content": "你好"}])  # type: ignore[list-item]


def test_factory_registers_and_creates_provider() -> None:
    """Factory 根据 Settings 创建已注册的 adapter。"""
    factory = _isolated_factory()
    factory.register_provider("fake", FakeLLM)
    settings = _settings_with_provider("fake")

    llm = factory.create(settings)

    assert isinstance(llm, FakeLLM)
    assert llm.settings == settings
    assert factory.list_providers() == ["fake"]


def test_factory_normalizes_provider_name() -> None:
    """注册和配置中的 provider 名称大小写不敏感。"""
    factory = _isolated_factory()
    factory.register_provider("FAKE", FakeLLM)
    settings = _settings_with_provider("fake")

    llm = factory.create(settings)

    assert isinstance(llm, FakeLLM)
    assert factory.list_providers() == ["fake"]


def test_factory_rejects_non_llm_class() -> None:
    """Registry 只接受实现 BaseLLM interface 的类。"""
    factory = _isolated_factory()

    class NotAnLLM:
        pass

    with pytest.raises(ValueError, match="must inherit from BaseLLM"):
        factory.register_provider("invalid", NotAnLLM)  # type: ignore[arg-type]


def test_factory_reports_unknown_provider() -> None:
    """未知 provider 错误包含请求值和可用列表。"""
    factory = _isolated_factory()
    factory.register_provider("fake", FakeLLM)
    settings = _settings_with_provider("unknown")

    with pytest.raises(ValueError) as exc_info:
        factory.create(settings)

    message = str(exc_info.value)
    assert "Unsupported LLM provider: 'unknown'" in message
    assert "Available providers: fake" in message


def test_factory_lists_providers_in_stable_order() -> None:
    """Provider 列表不受注册顺序影响。"""
    factory = _isolated_factory()
    factory.register_provider("zebra", FakeLLM)
    factory.register_provider("alpha", FakeLLM)
    factory.register_provider("beta", FakeLLM)

    assert factory.list_providers() == ["alpha", "beta", "zebra"]


def test_factory_reports_provider_construction_failure() -> None:
    """Adapter 构造失败时，错误包含 provider 名和原始原因。"""
    factory = _isolated_factory()

    class BrokenLLM(BaseLLM):
        def __init__(self, settings=None, **kwargs):
            raise RuntimeError("boom")

        def chat(self, messages, trace=None, **kwargs):
            return ChatResponse(content="unreachable", model="broken")

    factory.register_provider("broken", BrokenLLM)
    settings = _settings_with_provider("broken")

    with pytest.raises(RuntimeError) as exc_info:
        factory.create(settings)

    message = str(exc_info.value)
    assert "Failed to instantiate LLM provider 'broken'" in message
    assert "boom" in message
