"""只替换模型服务边界，验证真实提示词构造、解析和候选排序。"""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.llm.base_llm import BaseLLM, ChatResponse
from src.libs.reranker.reranker_factory import RerankerFactory

pytestmark = pytest.mark.unit


class StubLLM(BaseLLM):
    """外部模型响应固定，不使用 SDK、Key 或网络。"""

    def __init__(self, answer, check_request=None):
        self.answer = answer
        self.check_request = check_request

    def chat(self, messages, trace=None, **kwargs):
        self.validate_messages(messages)
        if self.check_request is not None:
            self.check_request(messages, trace, kwargs)
        return ChatResponse(self.answer, "test-llm")


def enabled_settings():
    """仅在测试配置中开启 LLM 重排。"""
    settings = load_settings()
    return replace(settings, rerank=replace(settings.rerank, enabled=True, provider="llm"))


def test_factory_llm_reranker_reads_prompt_and_preserves_candidate_content(tmp_path):
    """按模型评分排序，但原检索分数、文本及元数据不被替换。"""
    prompt = tmp_path / "rerank.txt"
    prompt.write_text("测试规则：判断候选能否回答问题。", encoding="utf-8")
    candidates = [
        {"id": "a", "text": "数据库简介", "score": 0.9, "metadata": {"source": "a.md"}},
        {"id": "b", "text": "选择 Chroma 的原因", "score": 0.5, "metadata": {"source": "b.md"}},
    ]
    original = deepcopy(candidates)

    def check_request(messages, trace, kwargs):
        text = messages[0].content
        assert "测试规则" in text
        assert "为什么选择 Chroma？" in text
        assert "Passage ID: a" in text and "数据库简介" in text
        assert "Passage ID: b" in text and "选择 Chroma 的原因" in text
        assert kwargs == {"temperature": 0.0}

    llm = StubLLM('[{"passage_id":"a","score":1},{"passage_id":"b","score":3}]', check_request)
    reranker = RerankerFactory.create(enabled_settings(), llm=llm, prompt_path=str(prompt))
    result = reranker.rerank("为什么选择 Chroma？", candidates, temperature=0.0)
    assert result == [dict(candidates[1], rerank_score=3.0), dict(candidates[0], rerank_score=1.0)]
    assert candidates == original


@pytest.mark.parametrize("answer", [
    "not-json", "{}", "[]", '["bad"]',
    '[{"passage_id":"a"}]',
    '[{"passage_id":"a","score":"3"},{"passage_id":"b","score":1}]',
    '[{"passage_id":"a","score":true},{"passage_id":"b","score":1}]',
    '[{"passage_id":"a","score":NaN},{"passage_id":"b","score":1}]',
    '[{"passage_id":"a","score":Infinity},{"passage_id":"b","score":1}]',
    '[{"passage_id":"a","score":4},{"passage_id":"b","score":1}]',
    '[{"passage_id":"a","score":-1},{"passage_id":"b","score":1}]',
    '[{"passage_id":"a","score":1}]',
    '[{"passage_id":"a","score":1},{"passage_id":"unknown","score":3}]',
    '[{"passage_id":"a","score":1},{"passage_id":"a","score":3},{"passage_id":"b","score":1}]',
])
def test_invalid_llm_response_signals_failure_instead_of_losing_candidates(answer):
    """JSON、ID 完整性或默认 0–3 分数契约有误时，不能静默丢弃候选。"""
    from src.libs.reranker.llm_reranker import LLMRerankError

    reranker = RerankerFactory.create(enabled_settings(), llm=StubLLM(answer))
    with pytest.raises(LLMRerankError):
        reranker.rerank("问题", [{"id": "a", "text": "甲"}, {"id": "b", "text": "乙"}])


def test_upstream_markdown_json_wrapper_is_supported():
    """保留上游对完整 JSON 代码围栏的兼容，不尝试猜测普通解释文本。"""
    answer = '```json\n[{"passage_id":"a","score":1},{"passage_id":"b","score":3}]\n```'
    reranker = RerankerFactory.create(enabled_settings(), llm=StubLLM(answer))
    assert [item["id"] for item in reranker.rerank("问题", [{"id": "a", "text": "甲"}, {"id": "b", "text": "乙"}])] == ["b", "a"]


def test_single_candidate_skips_model_call_like_upstream():
    def forbidden(*args):
        pytest.fail("只有一个候选不应调用 LLM")

    candidates = [{"id": "a", "text": "唯一候选", "score": 0.7}]
    reranker = RerankerFactory.create(enabled_settings(), llm=StubLLM("unused", forbidden))
    assert reranker.rerank("问题", candidates) == candidates


@pytest.mark.parametrize("error_type", [TimeoutError, RuntimeError])
def test_llm_failure_signals_fallback_without_echoing_model_response(error_type):
    from src.libs.reranker.llm_reranker import LLMRerankError

    class FailingLLM(StubLLM):
        def chat(self, messages, trace=None, **kwargs):
            raise error_type("private-test-value")

    reranker = RerankerFactory.create(enabled_settings(), llm=FailingLLM("unused"))
    with pytest.raises(LLMRerankError) as error:
        reranker.rerank("问题", [{"id": "a", "text": "甲"}, {"id": "b", "text": "乙"}])
    assert error_type.__name__ in str(error.value)
    assert "private-test-value" not in str(error.value)


@pytest.mark.parametrize("candidates", [
    [{"id": "a", "text": "甲"}, {"id": "a", "text": "乙"}],
    [{"id": "", "text": "甲"}, {"id": "b", "text": "乙"}],
    [{"id": "a", "text": None}, {"id": "b", "text": "乙"}],
    [{"id": "a", "text": " "}, {"id": "b", "text": "乙"}],
])
def test_invalid_candidate_identity_or_text_never_reaches_llm(candidates):
    def forbidden(*args):
        pytest.fail("非法候选不应调用 LLM")

    reranker = RerankerFactory.create(enabled_settings(), llm=StubLLM("unused", forbidden))
    with pytest.raises(ValueError):
        reranker.rerank("问题", candidates)


def test_blank_prompt_is_rejected_during_construction(tmp_path):
    prompt = tmp_path / "blank.txt"
    prompt.write_text("  \n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prompt"):
        RerankerFactory.create(enabled_settings(), llm=StubLLM("unused"), prompt_path=str(prompt))


def test_fallback_ids_content_alias_and_ties_preserve_original_order():
    """没有 ID 时沿用 passage_i，content 可作正文，同分不随模型输出顺序改变。"""
    answer = '[{"passage_id":"passage_1","score":2},{"passage_id":"passage_0","score":2}]'
    reranker = RerankerFactory.create(enabled_settings(), llm=StubLLM(answer))
    result = reranker.rerank("问题", [{"content": "甲"}, {"content": "乙"}])
    assert result == [{"content": "甲", "rerank_score": 2.0}, {"content": "乙", "rerank_score": 2.0}]


def test_missing_prompt_is_reported_without_calling_llm(tmp_path):
    with pytest.raises(RuntimeError, match="prompt"):
        RerankerFactory.create(enabled_settings(), llm=StubLLM("unused"), prompt_path=str(tmp_path / "missing.txt"))


@pytest.mark.parametrize("query,candidates", [(" ", [{"text": "甲"}]), ("问题", []), ("问题", ["bad"])])
def test_common_input_contract_is_checked_before_llm(query, candidates):
    def forbidden(*args):
        pytest.fail("非法输入不应调用模型")

    reranker = RerankerFactory.create(enabled_settings(), llm=StubLLM("unused", forbidden))
    with pytest.raises(ValueError):
        reranker.rerank(query, candidates)


def test_missing_llm_response_object_is_reported_as_backend_failure():
    from src.libs.reranker.llm_reranker import LLMRerankError

    class MissingResponseLLM(StubLLM):
        def chat(self, messages, trace=None, **kwargs):
            return None

    reranker = RerankerFactory.create(enabled_settings(), llm=MissingResponseLLM("unused"))
    with pytest.raises(LLMRerankError):
        reranker.rerank("问题", [{"id": "a", "text": "甲"}, {"id": "b", "text": "乙"}])
