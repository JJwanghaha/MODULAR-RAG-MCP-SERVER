"""两种视觉 adapter 共用的图片处理实现，不向工厂导入可选图像 SDK。"""

import base64
import io
from pathlib import Path

from src.libs.llm.base_vision_llm import ImageInput


def image_bytes(image: ImageInput) -> bytes:
    """读取明确指定的载体，Base64 只接受原始编码字符串。"""
    if image.data is not None:
        return image.data
    if image.path is not None:
        return Path(image.path).read_bytes()
    return base64.b64decode(image.base64, validate=True)


def resize_image(image: ImageInput, max_size: tuple[int, int] | None) -> ImageInput:
    """复现上游：Base64 或无 Pillow 时不缩放，路径/bytes 超限才处理。"""
    if max_size is None:
        return image
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in max_size):
        raise ValueError("max_size must contain positive integers")
    if image.base64 is not None:
        return image
    try:
        from PIL import Image
    except ImportError:
        return image
    with Image.open(io.BytesIO(image_bytes(image))) as original:
        width, height = original.size
        if width <= max_size[0] and height <= max_size[1]:
            return image
        ratio = min(max_size[0] / width, max_size[1] / height)
        size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        with original.resize(size, Image.Resampling.LANCZOS) as resized:
            output = io.BytesIO()
            resized.save(output, format=original.format or "PNG")
        return ImageInput(data=output.getvalue(), mime_type=image.mime_type)
