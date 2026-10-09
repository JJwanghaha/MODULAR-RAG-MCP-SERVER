"""C5–C7 的 list[Chunk]→list[Chunk] interface 与共用文本模型实现。"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from src.core.settings import resolve_path
from src.core.types import Chunk
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    """处理切片，保持输入数量和顺序，不自行编码或入库。"""

    @abstractmethod
    def transform(self, chunks: list[Chunk], trace=None) -> list[Chunk]:
        """由具体清理/增强实现决定正文或 metadata 变化。"""
        raise NotImplementedError


class _TextTransform(BaseTransform):
    """两种文本增强共享惰性模型、提示词和顺序保持，非新增公开接口。"""

    def __init__(self, settings, config_name, prompt_name, llm=None, prompt_path=None):
        config = getattr(settings.ingestion, config_name, None) or {}
        self.settings = settings
        self.use_llm = config.get("use_llm", False)
        self._llm = llm
        self._prompt_path = resolve_path(prompt_path or f"config/prompts/{prompt_name}.txt")
        self._prompt = None

    @property
    def llm(self):
        """关闭时不创建后端；构造失败禁用增强，保留规则路径。"""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
            except Exception as exc:
                logger.warning("Text model initialization failed: %s", type(exc).__name__)
                self.use_llm = False
        return self._llm

    def _load_prompt(self):
        if self._prompt is None:
            self._prompt = self._prompt_path.read_text(encoding="utf-8")
        return self._prompt

    def transform(self, chunks, trace=None):
        """规则串行；启用模型时按上游最多 5 个 worker，结果顺序不变。"""
        if not chunks:
            return []
        parallel = bool(self.use_llm and self.llm)
        if parallel:
            with ThreadPoolExecutor(max_workers=min(5, len(chunks))) as pool:
                results = list(pool.map(lambda value: self._transform_one(value, trace), chunks))
        else:
            results = [self._transform_one(value, trace) for value in chunks]
        return results

    @abstractmethod
    def _transform_one(self, chunk, trace):
        raise NotImplementedError
