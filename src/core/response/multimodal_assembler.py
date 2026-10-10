"""E6：按关联 ID 读取已有图片，以 MCP ImageContent 返回；不重新描述图片。"""

import base64
import hashlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.core.settings import resolve_path
from src.ingestion.storage.image_storage import ImageStorage

logger = logging.getLogger(__name__)
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


@dataclass
class ImageReference:
    image_id: str
    file_path: str | None = None
    page: int | None = None
    caption: str | None = None


@dataclass
class ImageContent:
    image_id: str
    data: str
    mime_type: str
    caption: str | None = None

    def to_mcp_content(self):
        from mcp import types
        return types.ImageContent(type="image", data=self.data, mimeType=self.mime_type)


class MultimodalAssembler:
    """只返回管理根目录内的 PNG/JPEG/WebP；失效图片不阻断文本结果。"""

    def __init__(self, image_storage=None, max_images_per_result: int = 5, *, images_root="data/images",
                 image_index_path="data/db/image_index.db", max_images_total: int = 5,
                 max_image_bytes: int = 2 * 1024 * 1024):
        if min(max_images_per_result, max_images_total, max_image_bytes) < 1:
            raise ValueError("Image limits must be positive")
        self._image_storage = image_storage
        self.images_root = resolve_path(images_root)
        self.image_index_path = resolve_path(image_index_path)
        self.max_images_per_result, self.max_images_total = max_images_per_result, max_images_total
        self.max_image_bytes = max_image_bytes

    def extract_image_refs(self, result) -> list[ImageReference]:
        refs = {}
        images = result.metadata.get("images", [])
        if isinstance(images, list):
            for image in images:
                if isinstance(image, dict) and image.get("id"):
                    refs.setdefault(str(image["id"]), ImageReference(str(image["id"]), image.get("path"), image.get("page")))
        identifiers = result.metadata.get("image_refs", [])
        if isinstance(identifiers, str):
            identifiers = identifiers.split(",")
        if not isinstance(identifiers, (list, tuple)):
            identifiers = []
        for identifier in [*identifiers, *IMAGE_PLACEHOLDER_PATTERN.findall(result.text)]:
            if isinstance(identifier, str) and identifier.strip():
                refs.setdefault(identifier.strip(), ImageReference(identifier.strip()))
        return list(refs.values())[:self.max_images_per_result]

    def resolve_image_path(self, ref: ImageReference, collection: str | None = None) -> str | None:
        """索引优先，原始列表 path 为备用；不猜测文件路径或反序列化字符串字典。"""
        if self._image_storage is not None:
            registered = self._image_storage.get_image_path(ref.image_id)
        else:
            registered = ImageStorage.get_existing_image_path(ref.image_id, self.image_index_path, collection)
        for value in (registered, ref.file_path):
            if value:
                path = Path(value).resolve()
                if path.is_relative_to(self.images_root.resolve()) and path.is_file():
                    return str(path)
        return None

    def load_image(self, file_path: str) -> ImageContent | None:
        """有界读取并用 Pillow 校验实际格式，不按后缀把任意文件当图片。"""
        try:
            from PIL import Image
            path = Path(file_path).resolve()
            if not path.is_relative_to(self.images_root.resolve()):
                return None
            with path.open("rb") as handle:
                data = handle.read(self.max_image_bytes + 1)
            if not data or len(data) > self.max_image_bytes:
                return None
            with Image.open(io.BytesIO(data)) as image:
                mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(image.format)
                image.verify()
            if mime is None:
                return None
            return ImageContent(path.stem, base64.b64encode(data).decode("ascii"), mime)
        except Exception as exc:
            logger.warning("Image content unavailable: %s", type(exc).__name__)
            return None

    def assemble(self, results, collection: str | None = None) -> list:
        blocks, seen_ids, seen_content = [], set(), set()
        for result in results:
            for ref in self.extract_image_refs(result):
                if ref.image_id in seen_ids:
                    continue
                seen_ids.add(ref.image_id)
                try:
                    path = self.resolve_image_path(ref, collection)
                    image = self.load_image(path) if path else None
                    if image is None:
                        continue
                    digest = hashlib.sha256(base64.b64decode(image.data)).hexdigest()
                    if digest in seen_content:
                        continue
                    seen_content.add(digest)
                    blocks.append(image.to_mcp_content())
                    if len(blocks) >= self.max_images_total:
                        return blocks
                except Exception as exc:
                    logger.warning("Image reference unavailable: %s", type(exc).__name__)
        return blocks

    def has_images(self, result) -> bool:
        return bool(self.extract_image_refs(result))

    def count_images(self, results) -> int:
        """仅计关联引用数量，不代表文件实际可读或最终返回数量。"""
        return sum(len(self.extract_image_refs(result)) for result in results)
