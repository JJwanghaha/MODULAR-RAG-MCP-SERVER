"""读取重排提示词，通过 LLM 结构化评分重新排列已有候选。"""

import json
from math import isfinite
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.reranker.base_reranker import BaseReranker


class LLMRerankError(RuntimeError):
    """模型调用或评分格式失败，由后续 Core 决定回退。"""


class LLMReranker(BaseReranker):
    """沿用上游 passage_id/score 协议，保留原检索分数。"""

    def __init__(self, settings: Any, prompt_path=None, llm=None, **kwargs: Any) -> None:
        self.settings = settings
        self.llm = llm if llm is not None else LLMFactory.create(settings)
        path = resolve_path(prompt_path or "config/prompts/rerank.txt")
        try:
            self.prompt_template = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise LLMRerankError(f"Unable to read rerank prompt: {type(exc).__name__}") from exc
        if not self.prompt_template.strip():
            raise LLMRerankError("Rerank prompt cannot be empty")

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], trace=None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """返回附带 rerank_score 的候选副本，默认不截断。"""
        self.validate_query(query)
        self.validate_candidates(candidates)
        identifiers = [
            candidate.get("id", f"passage_{i}")
            for i, candidate in enumerate(candidates)
        ]
        seen = set()
        for identifier, candidate in zip(identifiers, candidates):
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                raise ValueError("Candidate IDs must be unique non-empty strings")
            seen.add(identifier)
            text = candidate.get("text", candidate.get("content", ""))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Candidate text must be a non-empty string")
        if len(candidates) == 1:
            return list(candidates)
        passages = "\n\n".join(
            f"Passage ID: {identifier}\nText: {candidate.get('text', candidate.get('content', ''))}"
            for identifier, candidate in zip(identifiers, candidates)
        )
        prompt = (
            f"{self.prompt_template}\n\nQuery: {query}\n\nPassages:\n{passages}\n\n"
            "请返回每个候选的 JSON 评分数组。"
        )
        try:
            response = self.llm.chat([Message("user", prompt)], trace=trace, **kwargs)
            response_text = response.content
        except Exception as exc:
            raise LLMRerankError(f"LLM rerank call failed: {type(exc).__name__}") from None
        scores = self._parse_scores(response_text, identifiers)
        result = [
            dict(candidate, rerank_score=scores[identifier])
            for identifier, candidate in zip(identifiers, candidates)
        ]
        return sorted(result, key=lambda candidate: candidate["rerank_score"], reverse=True)

    @staticmethod
    def _parse_scores(text: str, identifiers: list[str]) -> dict[str, float]:
        """严格映射原候选；评分范围与默认及自定义提示词均约定为 0–3。"""
        try:
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            parsed = json.loads(text.strip())
        except (ValueError, TypeError, AttributeError):
            raise LLMRerankError("LLM rerank response must be valid JSON") from None
        if not isinstance(parsed, list) or len(parsed) != len(identifiers):
            raise LLMRerankError("LLM rerank must score every candidate exactly once")
        expected = set(identifiers)
        scores = {}
        for item in parsed:
            if not isinstance(item, dict):
                raise LLMRerankError("LLM rerank items must be objects")
            identifier, score = item.get("passage_id"), item.get("score")
            if (
                not isinstance(identifier, str)
                or identifier not in expected
                or identifier in scores
            ):
                raise LLMRerankError("LLM rerank contains invalid, unknown or repeated ID")
            if (
                not isinstance(score, (int, float)) or isinstance(score, bool)
                or not isfinite(score) or not 0 <= score <= 3
            ):
                raise LLMRerankError("LLM rerank score must be a finite number from 0 to 3")
            scores[identifier] = float(score)
        return scores
