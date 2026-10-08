"""在外部评分器边界验证 Cross-Encoder 协议，不下载真实模型。"""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.reranker.reranker_factory import RerankerFactory

pytestmark = pytest.mark.unit


class StubScorer:
    """提供固定的推理输出，只替换真实模型的 predict。"""

    def __init__(self, scores, check_pairs=None):
        self.scores = scores
        self.check_pairs = check_pairs

    def predict(self, pairs):
        if self.check_pairs is not None:
            self.check_pairs(pairs)
        return self.scores


def enabled_settings():
    settings = load_settings()
    return replace(settings, rerank=replace(settings.rerank, enabled=True, provider="cross_encoder"))


def test_factory_cross_encoder_predicts_pairs_and_limits_sorted_candidates():
    """每个问题/候选文本成对评分，允许负分，TopK 不改变原始记录。"""
    candidates = [
        {"id": "a", "text": "通用介绍", "score": 0.9},
        {"id": "b", "content": "具体原因", "score": 0.5, "metadata": {"source": "b.md"}},
        {"id": "c", "text": "补充说明", "score": 0.1},
    ]
    original = deepcopy(candidates)

    def check_pairs(pairs):
        assert pairs == [("为什么？", "通用介绍"), ("为什么？", "具体原因"), ("为什么？", "补充说明")]

    model = StubScorer([-2.0, 8.0, 1.0], check_pairs)
    reranker = RerankerFactory.create(enabled_settings(), model=model)
    result = reranker.rerank("为什么？", candidates, top_k=2)
    assert result == [dict(candidates[1], rerank_score=8.0), dict(candidates[2], rerank_score=1.0)]
    assert candidates == original


@pytest.mark.parametrize("scores", [[], [1.0], [1.0, 2.0, 3.0], [True, 1.0], ["1", 2], [float("nan"), 1], [float("inf"), 1], [[1.0], [2.0]]])
def test_invalid_score_shape_or_value_cannot_drop_candidates(scores):
    """每个候选必须有一个有限数值，不能依靠 zip 静默截断。"""
    from src.libs.reranker.cross_encoder_reranker import CrossEncoderRerankError

    reranker = RerankerFactory.create(enabled_settings(), model=StubScorer(scores))
    with pytest.raises(CrossEncoderRerankError):
        reranker.rerank("问题", [{"id": "a", "text": "甲"}, {"id": "b", "text": "乙"}])


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_is_rejected_before_prediction(top_k):
    def forbidden(pairs):
        pytest.fail("非法 TopK 不应调用模型")

    reranker = RerankerFactory.create(enabled_settings(), model=StubScorer([1, 2], forbidden))
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("问题", [{"text": "甲"}, {"text": "乙"}], top_k=top_k)


@pytest.mark.parametrize("error_type", [TimeoutError, RuntimeError])
def test_backend_failure_is_a_fallback_signal_without_echoing_private_text(error_type):
    from src.libs.reranker.cross_encoder_reranker import CrossEncoderRerankError

    class FailingScorer:
        def predict(self, pairs):
            raise error_type("private-test-value")

    reranker = RerankerFactory.create(enabled_settings(), model=FailingScorer())
    with pytest.raises(CrossEncoderRerankError) as error:
        reranker.rerank("问题", [{"text": "甲"}, {"text": "乙"}])
    assert error_type.__name__ in str(error.value)
    assert "private-test-value" not in str(error.value)


def test_array_like_scores_are_supported_and_ties_keep_original_order():
    """兼容模型 ndarray 的 tolist 协议，同分时保留检索顺序。"""
    class Scores:
        def tolist(self):
            return [2.0, 2.0]

    candidates = [{"id": "a", "text": "甲"}, {"id": "b", "text": "乙"}]
    reranker = RerankerFactory.create(enabled_settings(), model=StubScorer(Scores()))
    assert [item["id"] for item in reranker.rerank("问题", candidates)] == ["a", "b"]


@pytest.mark.parametrize("text", [None, 42, " "])
def test_invalid_candidate_text_is_rejected_before_prediction(text):
    def forbidden(pairs):
        pytest.fail("非法文本不应执行 predict")

    reranker = RerankerFactory.create(enabled_settings(), model=StubScorer([1], forbidden))
    with pytest.raises(ValueError, match="text"):
        reranker.rerank("问题", [{"text": text}])


def test_real_loader_requires_local_files_instead_of_downloading(monkeypatch):
    """在外部 SDK 构造边界验证本地加载策略，不安装 SDK 或访问模型站点。"""
    import sys
    from types import SimpleNamespace

    def load_model(name, *, local_files_only, trust_remote_code):
        assert name == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert local_files_only is True
        assert trust_remote_code is False
        return StubScorer([1.0, 2.0])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=load_model))
    reranker = RerankerFactory.create(enabled_settings())
    result = reranker.rerank("问题", [{"id": "a", "text": "甲"}, {"id": "b", "text": "乙"}])
    assert [item["id"] for item in result] == ["b", "a"]


@pytest.mark.parametrize("query,candidates", [(" ", [{"text": "甲"}]), ("问题", []), ("问题", ["bad"])])
def test_common_input_contract_is_checked_before_prediction(query, candidates):
    def forbidden(pairs):
        pytest.fail("非法输入不应执行 predict")

    reranker = RerankerFactory.create(enabled_settings(), model=StubScorer([1], forbidden))
    with pytest.raises(ValueError):
        reranker.rerank(query, candidates)
