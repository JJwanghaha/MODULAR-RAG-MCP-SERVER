"""统一 Loader interface：只解析，不切分、不编码、不调用模型。"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.core.types import Document


class BaseLoader(ABC):
    """不同格式输出同一 Document，沿用上游 load(file_path) 入口。"""

    @abstractmethod
    def load(self, file_path: str | Path) -> Document:
        """解析一份本地文档，保留来源。"""
        raise NotImplementedError

    @staticmethod
    def _validate_file(file_path: str | Path) -> Path:
        """只接受存在的本地文件，不把 URL 当作远程读取入口。"""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path
