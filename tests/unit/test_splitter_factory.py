"""通过公开接口验证切分策略的选择和文本契约。"""

from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

pytestmark = pytest.mark.unit


class FakeSplitter(BaseSplitter):
    """按空白切分的测试策略，不代表正式文档切分算法。"""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def split_text(self, text, trace=None, **kwargs):
        self.validate_text(text)
        chunks = text.split()
        self.validate_chunks(chunks)
        return chunks


@pytest.fixture
def factory():
    """隔离各测试的注册表，避免污染正式工厂。"""
    class TestFactory(SplitterFactory):
        _PROVIDERS = {}

    return TestFactory


def settings_for(strategy):
    """只修改测试关注的切分策略。"""
    settings = load_settings()
    return replace(settings, ingestion=replace(settings.ingestion, splitter=strategy))


def test_factory_creates_splitter_and_returns_ordered_text(factory):
    """配置选出的切分器接收 Settings，并返回有序文本。"""
    settings = settings_for("fake")
    factory.register_provider("fake", FakeSplitter)

    splitter = factory.create(settings, chunk_size=20)

    assert splitter.settings is settings
    assert splitter.kwargs == {"chunk_size": 20}
    assert splitter.split_text("hello world") == ["hello", "world"]
    assert factory.list_providers() == ["fake"]


@pytest.mark.parametrize("text", ["", "   ", 42])
def test_splitter_rejects_invalid_text(text):
    """空文本和非字符串在公开调用前被拒绝。"""
    with pytest.raises(ValueError):
        FakeSplitter().validate_text(text)


@pytest.mark.parametrize("chunks", [[], [""], ["ok", 42], "not-a-list"])
def test_splitter_rejects_invalid_output(chunks):
    """切分结果必须是非空字符串列表。"""
    with pytest.raises(ValueError):
        FakeSplitter().validate_chunks(chunks)


def test_splitter_checks_input_during_split():
    """具体策略调用 split_text 时也遵守输入契约。"""
    with pytest.raises(ValueError, match="cannot be empty"):
        FakeSplitter().split_text("  ")


def test_unknown_strategy_reports_available_names(factory):
    """配置中的未知策略返回可读错误。"""
    factory.register_provider("fake", FakeSplitter)
    with pytest.raises(ValueError, match="Unsupported Splitter provider: 'missing'"):
        factory.create(settings_for("missing"))


def test_missing_ingestion_configuration_reports_path(factory):
    """上游可选的 ingestion 未配置时，工厂说明缺失路径。"""
    settings = replace(load_settings(), ingestion=None)
    with pytest.raises(ValueError, match="settings.ingestion.splitter"):
        factory.create(settings)


def test_registration_rejects_incompatible_class(factory):
    """不能把未实现切分接口的类注册为策略。"""
    with pytest.raises(ValueError, match="must inherit from BaseSplitter"):
        factory.register_provider("bad", object)


def test_construction_failure_names_strategy(factory):
    """策略初始化失败时，工厂报告具体名称。"""
    class BrokenSplitter(FakeSplitter):
        def __init__(self, **kwargs):
            raise RuntimeError("failed")

    factory.register_provider("broken", BrokenSplitter)
    with pytest.raises(RuntimeError, match="Failed to instantiate Splitter provider 'broken'"):
        factory.create(settings_for("broken"))


def test_configuration_selects_different_strategies(factory):
    """配置切换后执行另一种策略，调用接口保持一致。"""
    class LineSplitter(FakeSplitter):
        def split_text(self, text, trace=None, **kwargs):
            self.validate_text(text)
            chunks = text.splitlines()
            self.validate_chunks(chunks)
            return chunks

    factory.register_provider("words", FakeSplitter)
    factory.register_provider("lines", LineSplitter)
    assert factory.create(settings_for("WORDS")).split_text("a b\nc") == ["a", "b", "c"]
    assert factory.create(settings_for("LINES")).split_text("a b\nc") == ["a b", "c"]
    assert factory.list_providers() == ["lines", "words"]
