"""C6 复现上游标题、摘要、标签规则和 Title/Summary/Tags 模型协议。"""

import logging
import re

from src.core.types import Chunk
from src.libs.llm.base_llm import Message
from src.ingestion.transform.base_transform import _TextTransform

logger = logging.getLogger(__name__)


class MetadataEnricher(_TextTransform):
    """正文不变，标题/摘要/标签按规则或模型结果覆盖相应 metadata。"""

    def __init__(self, settings, llm=None, prompt_path=None):
        super().__init__(settings, "metadata_enricher", "metadata_enrichment", llm, prompt_path)

    def _rule_metadata(self, text):
        if text is None:
            raise TypeError("Chunk text cannot be None")
        title = "Untitled"
        if text:
            heading = re.match(r"^#{1,6}\s+(.+)$", text, re.M)
            first = text.split("\n")[0].strip()
            if heading:
                title = heading.group(1).strip()
            elif first and len(first) <= 100 and not first.endswith((".", ",", ";")):
                title = first
            else:
                first_sentence = re.split(r"[.!?]\s+", text)[0].strip()
                title = re.sub(r"[.!?]+$", "", first_sentence)
                if len(title) > 150:
                    title = title[:147] + "..."
        summary = " ".join(re.split(r"(?<=[.!?])\s+", text)[:3]).strip()
        if len(summary) > 500:
            summary = summary[:497] + "..."
        tags = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)[:5])
        tags.update(re.findall(r"\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b", text)[:5])
        for groups in re.findall(r"\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_", text)[:5]:
            tags.update(value.strip() for value in groups if value)
        return {"title": title, "summary": summary, "tags": sorted(tags)[:10]}

    def _parse_metadata(self, text):
        title = re.search(r"Title:\s*(.+?)(?:\n|$)", text, re.I)
        summary = re.search(r"Summary:\s*(.+?)(?:\n(?:Tags:|$))", text, re.I | re.S)
        tags = re.search(r"Tags:\s*(.+?)(?:\n|$)", text, re.I)
        return {
            "title": title.group(1).strip() if title else "Untitled",
            "summary": summary.group(1).strip() if summary else text[:500],
            "tags": [value.strip() for value in tags.group(1).split(",") if value.strip()] if tags else [],
        }

    def _transform_one(self, chunk, trace):
        try:
            metadata = self._rule_metadata(chunk.text)
            method = "rule"
            if self.use_llm and self.llm:
                try:
                    prompt = self._load_prompt().replace("{chunk_text}", chunk.text[:2000])
                    response = self.llm.chat([Message("user", prompt)], trace=trace)
                    if not response:
                        raise ValueError("Empty response")
                    text = response.content if hasattr(response, "content") else response
                    metadata = self._parse_metadata(text)
                    method = "llm"
                except Exception as exc:
                    logger.warning("Metadata model failed: %s", type(exc).__name__)
                    metadata["enrich_fallback_reason"] = "llm_failed"
            return Chunk(chunk.id, chunk.text, {**chunk.metadata, **metadata, "enriched_by": method}, source_ref=chunk.source_ref)
        except Exception as exc:
            logger.warning("Metadata enrichment failed: %s", type(exc).__name__)
            return Chunk(chunk.id, chunk.text or "", {
                **chunk.metadata, "title": "Untitled", "summary": (chunk.text or "")[:100],
                "tags": [], "enriched_by": "error", "enrich_error": type(exc).__name__,
            }, source_ref=chunk.source_ref)
