"""C5 复现上游规则清理与可选 LLM 去噪，不提前改成 Markdown 保真策略。"""

import logging
import re

from src.core.types import Chunk
from src.libs.llm.base_llm import Message
from src.ingestion.transform.base_transform import _TextTransform

logger = logging.getLogger(__name__)


class ChunkRefiner(_TextTransform):
    """减少提取噪声；输出 ID 不随清理后的正文重新计算。"""

    def __init__(self, settings, llm=None, prompt_path=None):
        super().__init__(settings, "chunk_refiner", "chunk_refinement", llm, prompt_path)

    def _rule_clean(self, text):
        if not text:
            return text
        if not text.strip():
            return ""
        blocks = []

        def preserve(match):
            blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(blocks) - 1}__"

        text = re.sub(r"```[\s\S]*?```", preserve, text)
        text = re.sub(r"─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}", "", text, flags=re.I | re.S)
        text = re.sub(r"─{10,}", "", text)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        for index, block in enumerate(blocks):
            text = text.replace(f"__CODE_BLOCK_{index}__", block)
        return text.strip()

    def _transform_one(self, chunk, trace):
        try:
            text = self._rule_clean(chunk.text)
            method = "rule"
            if self.use_llm and self.llm and text:
                try:
                    prompt = self._load_prompt()
                    if "{text}" not in prompt:
                        raise ValueError("Missing {text} placeholder")
                    response = self.llm.chat([Message("user", prompt.replace("{text}", text))], trace=trace)
                    refined = response if isinstance(response, str) else response.content
                    if refined and refined.strip():
                        text, method = refined.strip(), "llm"
                except Exception as exc:
                    logger.warning("LLM refinement failed: %s", type(exc).__name__)
            return Chunk(chunk.id, text, {**chunk.metadata, "refined_by": method}, source_ref=chunk.source_ref)
        except Exception as exc:
            logger.warning("Chunk refinement failed: %s", type(exc).__name__)
            return chunk
