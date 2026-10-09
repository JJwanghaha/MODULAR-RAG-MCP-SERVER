"""复现上游 MarkItDown PDF 文本路线，图像提取由 PyMuPDF 完成。"""

import logging
import mimetypes
from pathlib import Path

from src.core.settings import resolve_path
from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.file_integrity import _sha256_file

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """只解析本地 PDF；不配置 OCR、LLM 或第三方转换插件。"""

    def __init__(self, extract_images=True, image_storage_dir="data/images"):
        self.extract_images = extract_images
        self.image_storage_dir = resolve_path(image_storage_dir)

    def load(self, file_path: str | Path) -> Document:
        """转换为正文，保留来源和原始字节哈希，不创建向量或处理记录。"""
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")
        try:
            from markitdown import MarkItDown
        except ImportError:
            raise ImportError("MarkItDown is not installed; install .[loaders]") from None
        doc_hash = _sha256_file(path)
        try:
            from pdfminer.pdfdocument import PDFDocument
            from pdfminer.pdfparser import PDFParser

            with path.open("rb") as handle:
                PDFDocument(PDFParser(handle))
            result = MarkItDown(enable_plugins=False).convert_local(path)
            text = result.text_content
        except Exception as exc:
            raise RuntimeError(f"PDF parsing failed: {type(exc).__name__}") from None
        metadata = {"source_path": str(path), "doc_type": "pdf", "doc_hash": doc_hash}
        for line in text.splitlines():
            if line.startswith("# "):
                metadata["title"] = line[2:].strip()
                break
        if self.extract_images:
            text, images = self._extract_images(path, text, doc_hash)
            if images:
                metadata["images"] = images
        return Document(f"doc_{doc_hash[:16]}", text, metadata)

    def _extract_images(self, path, text, doc_hash):
        """上游简化定位：占位符追加在全文末尾，不声称恢复 PDF 版面。"""
        try:
            import pymupdf as fitz
        except ImportError:
            logger.warning("PyMuPDF unavailable; PDF image extraction skipped")
            return text, []
        images = []
        try:
            with fitz.open(path) as pdf:
                for page_index, page in enumerate(pdf):
                    for index, info in enumerate(page.get_images(full=True)):
                        try:
                            image = pdf.extract_image(info[0])
                            image_id = f"{doc_hash}_{page_index + 1}_{index + 1}"
                            directory = self.image_storage_dir / doc_hash
                            directory.mkdir(parents=True, exist_ok=True)
                            destination = directory / f"{image_id}.{image['ext']}"
                            destination.write_bytes(image["image"])
                            placeholder = f"[IMAGE: {image_id}]"
                            offset = len(text) + 1
                            text += f"\n{placeholder}\n"
                            images.append({
                                "id": image_id, "path": str(destination), "page": page_index + 1,
                                "text_offset": offset, "text_length": len(placeholder),
                                "mime_type": mimetypes.guess_type(destination.name)[0] or "application/octet-stream",
                                "position": {"width": image.get("width", 0), "height": image.get("height", 0),
                                             "page": page_index + 1, "index": index},
                            })
                        except Exception as exc:
                            logger.warning("PDF image skipped: %s", type(exc).__name__)
        except Exception as exc:
            logger.warning("PDF image extraction unavailable: %s", type(exc).__name__)
        return text, images
