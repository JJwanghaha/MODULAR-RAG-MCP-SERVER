"""验证关闭重排和切换策略时的公开行为。"""

from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory

pytestmark = pytest.mark.unit


@pytest.fixture
def factory():
    """为每次测试提供独立注册表。"""
    class TestFactory(RerankerFactory):
        _PROVIDERS = {}

    return TestFactory


def settings_for(provider, enabled=True):
    """只替换重排配置。"""
    settings = load_settings()
    return replace(settings, rerank=replace(settings.rerank, provider=provider, enabled=enabled))


def test_disabled_reranking_preserves_original_order(factory):
    """关闭重排后，统一调用仍返回候选原顺序和数据。"""
    reranker = factory.create(settings_for("not-installed", enabled=False))
    candidates = [{"id": "a", "score": 0.2}, {"id": "b", "score": 0.9}]
    results = reranker.rerank("查询", candidates)
    assert isinstance(reranker, NoneReranker)
    assert results == candidates
    assert results is not candidates


@pytest.mark.parametrize("query,candidates", [
    ("", [{"id": "a"}]), ("  ", [{"id": "a"}]), (42, [{"id": "a"}]),
    ("查询", []), ("查询", "bad"), ("查询", ["bad"]),
])
def test_none_reranker_respects_input_contract(query, candidates):
    """无操作策略同样验证问题和候选格式。"""
    with pytest.raises(ValueError):
        NoneReranker().rerank(query, candidates)


class FakeReranker(BaseReranker):
    """仅用现有分数演示策略切换，不计算模型相关度。"""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def rerank(self, query, candidates, trace=None, **kwargs):
        self.validate_query(query)
        self.validate_candidates(candidates)
        return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)


def test_enabled_factory_selects_registered_strategy(factory):
    """开启重排后，配置选中的策略实际改变候选顺序。"""
    factory.register_provider("FAKE", FakeReranker)
    settings = settings_for("fake")
    reranker = factory.create(settings, top_k=2)
    candidates = [{"id": "a", "score": 0.2}, {"id": "b", "score": 0.9}]
    assert reranker.settings is settings
    assert reranker.kwargs == {"top_k": 2}
    assert reranker.rerank("查询", candidates) == [candidates[1], candidates[0]]
    assert [item["id"] for item in candidates] == ["a", "b"]
    assert factory.list_providers() == ["fake"]


def test_enabled_unknown_reranker_is_reported(factory):
    """开启状态下不能静默回退未知策略。"""
    with pytest.raises(ValueError, match="Unsupported Reranker provider: 'missing'"):
        factory.create(settings_for("missing"))


def test_reranker_registration_requires_base_interface(factory):
    """注册类必须遵守重排接口。"""
    with pytest.raises(ValueError, match="must inherit from BaseReranker"):
        factory.register_provider("bad", object)


def test_reranker_construction_failure_names_provider(factory):
    """错误定位到初始化失败的策略。"""
    class BrokenReranker(FakeReranker):
        def __init__(self, **kwargs):
            raise RuntimeError("failed")

    factory.register_provider("broken", BrokenReranker)
    with pytest.raises(RuntimeError, match="Failed to instantiate Reranker provider 'broken'"):
        factory.create(settings_for("broken"))


def test_none_provider_selects_noop_even_when_enabled(factory):
    """显式 none 是上游支持的另一条无操作路径。"""
    assert isinstance(factory.create(settings_for("NONE")), NoneReranker)
