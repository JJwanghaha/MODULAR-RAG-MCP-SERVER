"""C7 对完整图片占位符生成描述；按上游批内缓存、原块写回。"""

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.core.settings import resolve_path
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import ImageInput
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


class ImageCaptioner(BaseTransform):
    """只描述当前正文引用的图片，不负责下载、解析或向量化。"""

    def __init__(self, settings, llm=None):
        self.llm = None
        if settings.vision_llm and settings.vision_llm.enabled:
            try:
                self.llm = llm or LLMFactory.create_vision_llm(settings)
            except Exception as exc:
                logger.warning("Vision initialization failed: %s", type(exc).__name__)
        self._cache = {}
        self._lock = threading.Lock()
        path = resolve_path("config/prompts/image_captioning.txt")
        self.prompt = path.read_text(encoding="utf-8").strip() if path.exists() else "描述图片中可用于检索的实际内容。"

    def _caption(self, image_id, path, trace):
        with self._lock:
            if image_id in self._cache:
                return self._cache[image_id]
        if not path or not Path(path).exists():
            return None
        try:
            response = self.llm.chat_with_image(self.prompt, ImageInput(path=path), trace=trace)
            caption = response.content
            with self._lock:
                self._cache[image_id] = caption
            return caption
        except Exception as exc:
            logger.warning("Image caption failed: %s", type(exc).__name__)
            return None

    def transform(self, chunks, trace=None):
        """关闭时原样返回；每次调用重置缓存，不保证重复写回幂等。"""
        if not self.llm:
            return chunks
        lookup = {}
        for chunk in chunks:
            for image in chunk.metadata.get("images", []):
                if image.get("id"):
                    lookup.setdefault(image["id"], image)
        with self._lock:
            self._cache.clear()
        required = {}
        for chunk in chunks:
            for reference in IMAGE_PLACEHOLDER_PATTERN.findall(chunk.text):
                image_id = reference.strip()
                image = lookup.get(image_id)
                if image and image.get("path"):
                    required.setdefault(image_id, image["path"])
        if required:
            with ThreadPoolExecutor(max_workers=min(3, len(required))) as pool:
                list(pool.map(lambda item: self._caption(item[0], item[1], trace), required.items()))
        for chunk in chunks:
            captions = []
            for reference in IMAGE_PLACEHOLDER_PATTERN.findall(chunk.text):
                image_id = reference.strip()
                caption = self._cache.get(image_id)
                if caption:
                    placeholder = f"[IMAGE: {image_id}]"
                    chunk.text = chunk.text.replace(placeholder, f"{placeholder}\n(Description: {caption})")
                    captions.append({"id": image_id, "caption": caption})
            if captions:
                chunk.metadata.setdefault("image_captions", []).extend(captions)
        return list(chunks)
