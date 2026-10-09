"""C14：沿用上游六阶段摄取流程，补上 Markdown 路由。"""

import logging
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from src.core.settings import Settings, load_settings, resolve_path
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".markdown"})


@dataclass
class PipelineResult:
    """保留上游统计字段；doc_id 是原始文件 SHA256，不是 Loader 的短 ID。"""

    success: bool
    file_path: str
    doc_id: str | None = None
    chunk_count: int = 0
    image_count: int = 0
    vector_ids: list[str] = field(default_factory=list)
    error: str | None = None
    stages: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """不包含正文、向量内容或模型原始错误。"""
        return {"success": self.success, "file_path": self.file_path, "doc_id": self.doc_id,
                "chunk_count": self.chunk_count, "image_count": self.image_count,
                "vector_ids_count": len(self.vector_ids), "error": self.error, "stages": self.stages}


class IngestionPipeline:
    """单文件入口；一个实例串行处理同一集合，不提供跨存储事务。"""

    def __init__(self, settings: Settings, collection: str = "default", force: bool = False,
                 *, embedding: BaseEmbedding | None = None, data_dir: str | Path | None = None,
                 source_root: str | Path | None = None):
        if not collection.strip() or collection in {".", ".."} or any(char in collection for char in "/\\"):
            raise ValueError("collection must be a single path component")
        if settings.ingestion is None:
            raise ValueError("Ingestion configuration is required")
        self.data_dir = resolve_path(data_dir if data_dir is not None else "data")
        # 显式隔离目录时所有摄取存储放进该目录；默认仍尊重已有 Chroma 配置。
        self.settings = replace(settings, vector_store=replace(settings.vector_store,
                                persist_directory=str(self.data_dir / "db" / "chroma"))) if data_dir is not None else settings
        self.collection, self.force = collection, force
        self.source_root = Path(source_root).resolve() if source_root is not None else None
        self._embedding = embedding
        self.integrity_checker = SQLiteIntegrityChecker(str(self.data_dir / "db" / "ingestion_history.db"))
        self.chunker = None
        self.batch_processor = None
        self.vector_upserter = None
        self.image_storage = None
        self._closed = False

    def _initialize_processing(self) -> None:
        """通过跳过检查后才创建模型/编码器/存储；离线测试只替换 Embedding seam。"""
        if self.batch_processor is not None:
            return
        embedding = self._embedding if self._embedding is not None else EmbeddingFactory.create(self.settings)
        batch_size = self.settings.ingestion.batch_size
        self.chunker = DocumentChunker(self.settings)
        self.chunk_refiner = ChunkRefiner(self.settings)
        self.metadata_enricher = MetadataEnricher(self.settings)
        self.image_captioner = ImageCaptioner(self.settings)
        processor = BatchProcessor(DenseEncoder(embedding, batch_size), SparseEncoder(), batch_size)
        self.bm25_indexer = BM25Indexer(str(self.data_dir / "db" / "bm25" / self.collection))
        self.image_storage = ImageStorage(str(self.data_dir / "db" / "image_index.db"), str(self.data_dir / "images"))
        # 持有长期连接的 Chroma 最后创建，避免后续初始化失败后重试遗留客户端。
        self.vector_upserter = VectorUpserter(self.settings, self.collection)
        self.batch_processor = processor

    def run(self, file_path: str | Path, trace: Any = None,
            on_progress: Callable[[str, int, int], None] | None = None) -> PipelineResult:
        """解析、增强、编码和存储；错误用阶段＋类型记录，不暴露私有正文或 Key。"""
        path = Path(file_path).resolve()
        stages: dict[str, Any] = {}
        file_hash = None
        stage = "integrity"
        document = None
        chunks, vector_ids = [], []

        def notify(name: str, step: int) -> None:
            if on_progress is not None:
                try:
                    on_progress(name, step, 6)
                except Exception as exc:
                    # 显示进度失败不能把已完成的持久化工作误标为业务失败。
                    logger.warning("Progress callback failed: %s", type(exc).__name__)

        try:
            if self._closed:
                raise RuntimeError("Pipeline is closed")
            if not path.is_file():
                raise FileNotFoundError("Source must be an existing file")
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError("Unsupported file type")
            file_hash = self.integrity_checker.compute_sha256(str(path))
            skipped = not self.force and self.integrity_checker.should_skip(file_hash)
            stages["integrity"] = {"file_hash": file_hash, "skipped": skipped}
            notify("integrity", 1)
            if skipped:
                return PipelineResult(True, str(path), file_hash, stages=stages)

            stage = "initialization"
            self._initialize_processing()
            stage = "loading"
            image_dir = self.data_dir / "images" / self.collection
            loader = (PdfLoader(extract_images=True, image_storage_dir=image_dir) if path.suffix.lower() == ".pdf"
                      else MarkdownLoader(image_storage_dir=image_dir, source_root=self.source_root))
            document = loader.load(path)
            # 防止读取期间文件变化，却给另一版本的内容标记 success。
            if document.metadata["doc_hash"] != file_hash:
                raise ValueError("Source changed during loading; retry ingestion")
            images = document.metadata.get("images", [])
            stages["loading"] = {"doc_id": document.id, "doc_type": document.metadata["doc_type"],
                                 "text_length": len(document.text), "image_count": len(images)}
            notify("load", 2)

            stage = "chunking"
            chunks = self.chunker.split_document(document)
            stages["chunking"] = {"chunk_count": len(chunks)}
            notify("split", 3)

            stage = "transform"
            chunks = self.chunk_refiner.transform(chunks, trace=trace)
            chunks = self.metadata_enricher.transform(chunks, trace=trace)
            chunks = self.image_captioner.transform(chunks, trace=trace)
            stages["transform"] = {"chunk_count": len(chunks),
                                   "refined_by_llm": sum(c.metadata.get("refined_by") == "llm" for c in chunks),
                                   "enriched_by_llm": sum(c.metadata.get("enriched_by") == "llm" for c in chunks),
                                   "captioned_chunks": sum(bool(c.metadata.get("image_captions")) for c in chunks)}
            notify("transform", 4)

            stage = "encoding"
            encoded = self.batch_processor.process(chunks, trace=trace)
            stages["encoding"] = {"dense_vector_count": len(encoded.dense_vectors),
                                  "sparse_doc_count": len(encoded.sparse_stats), "batch_count": encoded.batch_count,
                                  "dense_dimension": len(encoded.dense_vectors[0])}
            notify("embed", 5)

            stage = "storage"
            stages["storage"] = {"vector_count": 0, "bm25_docs": 0, "images_indexed": 0}
            vector_ids = self.vector_upserter.upsert(chunks, encoded.dense_vectors, trace=trace)
            stages["storage"]["vector_count"] = len(vector_ids)
            sparse_stats = [{**stat, "chunk_id": vid} for stat, vid in zip(encoded.sparse_stats, vector_ids)]
            # 上游传 document.id，但最终 ID 不以它开头；改用最终 ID 对应的来源哈希前缀。
            source_prefix = sha256(str(path).encode("utf-8")).hexdigest()[:8] + "_"
            self.bm25_indexer.add_documents(sparse_stats, self.collection, doc_id=source_prefix, trace=trace)
            stages["storage"]["bm25_docs"] = len(sparse_stats)
            for image in images:
                self.image_storage.register_image(image["id"], image["path"], self.collection,
                                                  file_hash, image.get("page"))
                stages["storage"]["images_indexed"] += 1
            notify("upsert", 6)
            stage = "completion"
            self.integrity_checker.mark_success(file_hash, str(path), self.collection)
            return PipelineResult(True, str(path), file_hash, len(chunks), len(images), vector_ids, stages=stages)
        except Exception as exc:
            error = f"{stage} failed: {type(exc).__name__}"
            stages["failure"] = {"stage": stage, "error_type": type(exc).__name__}
            logger.error("Ingestion %s", error)
            if file_hash is not None and not self._closed:
                try:
                    self.integrity_checker.mark_failed(file_hash, str(path), error)
                except Exception as history_exc:
                    stages["failure"]["history_error_type"] = type(history_exc).__name__
                    logger.error("Failure history could not be saved: %s", type(history_exc).__name__)
            return PipelineResult(False, str(path), file_hash, len(chunks),
                                  stages.get("storage", {}).get("images_indexed", 0), vector_ids, error, stages)

    def close(self) -> None:
        """关闭流水线持有的本地存储；可重复调用，不关闭注入的模型。"""
        if self._closed:
            return
        try:
            if self.vector_upserter is not None:
                client = getattr(self.vector_upserter.vector_store, "client", None)
                if client is not None:
                    client.close()
        finally:
            if self.image_storage is not None:
                self.image_storage.close()
            self._closed = True


def run_pipeline(file_path: str | Path, settings_path: str | Path | None = None,
                 collection: str = "default", force: bool = False) -> PipelineResult:
    """单次调用便捷入口，结束时确保关闭存储。"""
    pipeline = IngestionPipeline(load_settings(settings_path), collection, force)
    try:
        return pipeline.run(file_path)
    finally:
        pipeline.close()
