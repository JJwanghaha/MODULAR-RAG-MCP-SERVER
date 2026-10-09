"""视觉模型的图片输入与统一 interface，沿用上游单图约定。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.libs.llm.base_llm import ChatResponse, Message


@dataclass
class ImageInput:
    """图片来源三选一；这里只描述输入，不读取或解码图片。"""

    path: str | Path | None = None
    data: bytes | None = None
    base64: str | None = None
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        """与上游一致，按非 None 字段检查来源数量。"""
        count = sum(value is not None for value in (self.path, self.data, self.base64))
        if count == 0:
            raise ValueError("Must provide one of: path, data, or base64")
        if count > 1:
            raise ValueError("Must provide exactly one of: path, data, or base64")


class BaseVisionLLM(ABC):
    """独立于文本 LLM 的单图 interface，不限制底层使用同一个模型。"""

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: list[Message] | None = None,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """结合文字与一张图片生成响应；历史、Trace 及选项由 adapter 处理。"""
        raise NotImplementedError

    def validate_text(self, text: str) -> None:
        """校验调用提示词；具体 adapter 须在请求前调用。"""
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Text prompt cannot be empty")

    def validate_image(self, image: ImageInput) -> None:
        """只检查统一输入类型，不读取文件或判断图像有效性。"""
        if not isinstance(image, ImageInput):
            raise ValueError(f"Image must be an ImageInput instance, got {type(image).__name__}")

    def preprocess_image(
        self, image: ImageInput, max_size: tuple[int, int] | None = None
    ) -> ImageInput:
        """上游的预处理扩展点；默认原样返回，压缩由具体 adapter 实现。"""
        return image
