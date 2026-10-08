"""用已知答案验证轻量检索指标和评估工厂。"""

from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.evaluator.base_evaluator import NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory

pytestmark = pytest.mark.unit


@pytest.fixture
def factory():
    """独立注册表保留上游默认 custom 提供者。"""
    class TestFactory(EvaluatorFactory):
        _PROVIDERS = {"custom": CustomEvaluator}

    return TestFactory


def settings_for(provider, enabled=True):
    """只替换评估开关和提供者。"""
    settings = load_settings()
    return replace(settings, evaluation=replace(settings.evaluation, provider=provider, enabled=enabled))


def test_first_relevant_result_at_second_position():
    """标准答案排第二：命中为 1，倒数排名为 1/2。"""
    evaluator = CustomEvaluator()
    metrics = evaluator.evaluate(
        "问题", [{"id": "a"}, {"id": "b"}, {"id": "c"}], ground_truth=["b", "z"]
    )
    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}


def test_requested_metrics_only():
    """只请求 mrr 时不返回其他指标。"""
    metrics = CustomEvaluator(metrics=["mrr"]).evaluate(
        "问题", [{"id": "a"}, {"id": "b"}], ground_truth=["b"]
    )
    assert metrics == {"mrr": 0.5}


def test_unsupported_metric_is_not_silently_reported():
    """轻量评估器不能声称计算了尚未实现的忠实度指标。"""
    with pytest.raises(ValueError, match="Unsupported custom metrics: faithfulness"):
        CustomEvaluator(metrics=["faithfulness"])


@pytest.mark.parametrize("query,chunks", [("", [{"id": "a"}]), (42, [{"id": "a"}]), ("问题", []), ("问题", "bad")])
def test_evaluator_rejects_invalid_input(query, chunks):
    """上游基线要求非空问题和非空候选列表。"""
    with pytest.raises(ValueError):
        CustomEvaluator().evaluate(query, chunks, ground_truth=["a"])


@pytest.mark.parametrize("chunks,truth", [
    (["a", "b"], "b"), ([{"chunk_id": "a"}, {"doc_id": "b"}], {"ids": ["b"]}),
    ([{"document_id": "a"}, {"id": "b"}], [{"id": "b"}]),
])
def test_upstream_identifier_shapes_are_supported(chunks, truth):
    """文本 ID 和上游各类记录标识都映射到同一评估逻辑。"""
    assert CustomEvaluator().evaluate("问题", chunks, ground_truth=truth) == {
        "hit_rate": 1.0, "mrr": 0.5
    }


def test_factory_selects_custom_evaluator(factory):
    """工厂选择内置轻量评估器，支持显式指标覆盖。"""
    evaluator = factory.create(settings_for("CUSTOM"), metrics=["mrr"])
    assert isinstance(evaluator, CustomEvaluator)
    assert evaluator.evaluate("问题", ["a", "b"], ground_truth=["b"]) == {"mrr": 0.5}


def test_metrics_are_read_from_settings(factory):
    """调用者未覆盖指标时遵循配置文件。"""
    settings = settings_for("custom")
    settings = replace(settings, evaluation=replace(settings.evaluation, metrics=["hit_rate"]))
    evaluator = factory.create(settings)
    assert evaluator.evaluate("问题", ["a", "b"], ground_truth=["b"]) == {"hit_rate": 1.0}


def test_default_template_can_enable_custom_evaluation(factory):
    """仅开启默认评估开关即可使用模板中声明的指标。"""
    settings = settings_for("custom")
    evaluator = factory.create(settings)
    assert evaluator.evaluate("问题", ["a"], ground_truth=["a"]) == {
        "hit_rate": 1.0, "mrr": 1.0
    }


def test_factory_rejects_unknown_evaluator(factory):
    """开启评估时，未知提供者不能静默返回指标。"""
    with pytest.raises(ValueError, match="Unsupported Evaluator provider: 'missing'"):
        factory.create(settings_for("missing"))


def test_registered_evaluator_is_selected_by_configuration(factory):
    """评估器工厂也支持新增策略并保持统一调用方式。"""
    class FakeEvaluator(CustomEvaluator):
        pass

    factory.register_provider("FAKE", FakeEvaluator)
    evaluator = factory.create(settings_for("fake"), metrics=["mrr"])
    assert isinstance(evaluator, FakeEvaluator)
    assert evaluator.evaluate("问题", ["a"], ground_truth=["a"]) == {"mrr": 1.0}
    assert factory.list_providers() == ["custom", "fake"]


def test_evaluator_registration_requires_base_interface(factory):
    """评估工厂拒绝未实现接口的类。"""
    with pytest.raises(ValueError, match="must inherit from BaseEvaluator"):
        factory.register_provider("bad", object)


def test_evaluator_construction_failure_names_provider(factory):
    """初始化失败定位到选中的评估器。"""
    class BrokenEvaluator(CustomEvaluator):
        def __init__(self, **kwargs):
            raise RuntimeError("failed")

    factory.register_provider("broken", BrokenEvaluator)
    with pytest.raises(RuntimeError, match="Failed to instantiate Evaluator provider 'broken'"):
        factory.create(settings_for("broken"))


@pytest.mark.parametrize("provider,enabled", [("not-installed", False), ("none", True), ("disabled", True)])
def test_disabled_evaluation_returns_empty_metrics(factory, provider, enabled):
    """关闭或显式 none/disabled 时不计算指标。"""
    evaluator = factory.create(settings_for(provider, enabled))
    assert isinstance(evaluator, NoneEvaluator)
    assert evaluator.evaluate("问题", ["a"], ground_truth=["a"]) == {}


@pytest.mark.parametrize("truth,expected", [
    (["a"], {"hit_rate": 1.0, "mrr": 1.0}),
    (["c"], {"hit_rate": 1.0, "mrr": 1.0 / 3}),
    (["c", "b"], {"hit_rate": 1.0, "mrr": 0.5}),
    (["z"], {"hit_rate": 0.0, "mrr": 0.0}),
    ([], {"hit_rate": 0.0, "mrr": 0.0}),
    (None, {"hit_rate": 0.0, "mrr": 0.0}),
])
def test_known_rank_examples(truth, expected):
    """已知排名独立验证首个命中、未命中和上游无标签约定。"""
    assert CustomEvaluator().evaluate("问题", ["a", "b", "c"], ground_truth=truth) == expected


@pytest.mark.parametrize("chunks,truth,error", [
    ([{}], ["a"], "Missing id field"),
    ([42], ["a"], "Unable to extract id"),
    (["a"], 42, "Unsupported ground_truth type"),
])
def test_invalid_identifiers_are_reported(chunks, truth, error):
    """无法定位标准答案时返回明确错误。"""
    with pytest.raises(ValueError, match=error):
        CustomEvaluator().evaluate("问题", chunks, ground_truth=truth)
