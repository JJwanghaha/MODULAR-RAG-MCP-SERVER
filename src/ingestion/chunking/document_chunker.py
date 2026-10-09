"""C4 将已有字符串切分结果包装成上游 Chunk 对象。"""

import hashlib
import re

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory


class DocumentChunker:
    """复用配置指定的切分器，不读取文件、不编码或写入索引。"""

    def __init__(self, settings):
        self._splitter = SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        """补齐稳定 ID、顺序与文档来源；独立偏移字段保持上游默认值。"""
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")
        fragments = self._splitter.split_text(document.text)
        if not fragments:
            raise ValueError(f"Splitter returned no chunks for document {document.id}. Text length: {len(document.text)}")
        chunks = []
        for index, text in enumerate(fragments):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
            metadata = document.metadata.copy()
            document_images = metadata.pop("images", [])
            metadata["chunk_index"] = index
            metadata["source_ref"] = document.id
            image_refs = [value.strip() for value in re.findall(r"\[IMAGE:\s*([^\]]+)\]", text)]
            metadata["image_refs"] = image_refs
            lookup = {image.get("id"): image for image in document_images} if image_refs and document_images else {}
            images = [lookup[image_id] for image_id in image_refs if image_id in lookup]
            if images:
                metadata["images"] = images
                metadata["page_num"] = images[0].get("page")
            chunks.append(Chunk(f"{document.id}_{index:04d}_{digest}", text, metadata))
        return chunks
