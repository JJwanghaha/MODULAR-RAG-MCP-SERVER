"""E5：取已有文档的块元数据摘要或正文预览，不调用生成模型。"""

import asyncio
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.mcp_server.tools._storage import configured_storage, existing_chroma_client, validate_collection

TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = "按文档 ID 获取标题、已有摘要/正文预览、标签、来源和块数；不重新调用模型生成摘要。"
TOOL_INPUT_SCHEMA = {"type": "object", "properties": {
    "doc_id": {"type": "string", "minLength": 1, "description": "引用 metadata.source_ref 的 doc_短哈希，或完整 doc_hash"},
    "collection": {"type": "string", "minLength": 1, "description": "集合名，默认 default"}},
    "required": ["doc_id"], "additionalProperties": False}


@dataclass
class DocumentSummary:
    doc_id: str
    title: str
    summary: str
    tags: list[str] = field(default_factory=list)
    source_path: str | None = None
    chunk_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GetDocumentSummaryConfig:
    default_collection: str = "default"
    summary_max_length: int = 500


class DocumentNotFoundError(LookupError):
    """集合或指定文档不存在；不把输入、内部路径或存储异常复制到错误正文。"""


class GetDocumentSummaryTool:
    def __init__(self, settings=None, config=None, *, data_dir=None):
        self.settings, self.data_root = configured_storage(settings, data_dir)
        self.config = config if config is not None else GetDocumentSummaryConfig()

    def get_document_summary(self, doc_id: str, collection: str | None = None) -> DocumentSummary:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be nonblank")
        selected = self.config.default_collection if collection is None else collection
        validate_collection(selected)
        if re.fullmatch(r"[0-9a-fA-F]{64}", doc_id):
            where = {"doc_hash": doc_id.lower()}
        elif re.fullmatch(r"[0-9a-fA-F]{16}", doc_id):
            where = {"source_ref": "doc_" + doc_id.lower()}
        else:
            where = {"source_ref": doc_id}
        try:
            with existing_chroma_client(self.settings) as client:
                import chromadb
                try:
                    stored = client.get_collection(selected, embedding_function=None)
                except chromadb.errors.NotFoundError:
                    raise DocumentNotFoundError("Document collection not found") from None
                result = stored.get(where=where, include=["documents", "metadatas"])
        except FileNotFoundError:
            raise DocumentNotFoundError("Knowledge database not found") from None
        chunks = [{"id": cid, "text": text or "", "metadata": metadata or {}}
                  for cid, text, metadata in zip(result["ids"], result["documents"], result["metadatas"])]
        if not chunks:
            raise DocumentNotFoundError("Document not found")
        chunks.sort(key=lambda item: (item["metadata"].get("chunk_index", 0), item["id"]))
        first = chunks[0]
        metadata, text = first["metadata"], first["text"]
        source = metadata.get("source_path")
        title = metadata.get("title")
        if not title:
            title = next((line[2:].strip() for line in text.splitlines()[:10] if line.startswith("# ")), None)
        title = str(title or (Path(source).stem if source else "未命名文档"))
        summary = next((str(item["metadata"]["summary"]) for item in chunks if item["metadata"].get("summary")), "")
        summary_kind = "stored_chunk_summary" if summary else "excerpt"
        if not summary:
            summary = " ".join(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))
        summary = summary[:self.config.summary_max_length] or "暂无正文预览。"
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        if not isinstance(tags, (list, tuple)):
            tags = []
        tags = list(dict.fromkeys(str(tag) for tag in tags))
        if metadata.get("doc_type"):
            kind = str(metadata["doc_type"]).upper()
            if kind not in tags:
                tags.append(kind)
        return DocumentSummary(doc_id, title, summary, tags, source, len(chunks),
                               {**{key: metadata[key] for key in ("doc_type", "doc_hash", "source_ref") if key in metadata},
                                "summary_kind": summary_kind})

    def format_response(self, summary: DocumentSummary) -> str:
        return (f"## 文档：{summary.title}\n\n文档 ID：`{summary.doc_id}`\n\n来源：`{summary.source_path or 'unknown'}`"
                f"\n\n块数：{summary.chunk_count}\n\n标签：{', '.join(summary.tags)}\n\n### 摘要/预览\n\n{summary.summary}")

    async def execute(self, doc_id: str, collection: str | None = None):
        from mcp import types
        try:
            summary = await asyncio.to_thread(self.get_document_summary, doc_id, collection)
        except DocumentNotFoundError:
            return types.CallToolResult(content=[types.TextContent(type="text", text="未找到文档，请检查 doc_id 和集合名。")],
                                        isError=True)
        return types.CallToolResult(content=[types.TextContent(type="text", text=self.format_response(summary))],
                                    structuredContent=summary.to_dict(), isError=False)


def register_tool(protocol_handler, settings=None, *, data_dir=None) -> None:
    tool = GetDocumentSummaryTool(settings, data_dir=data_dir)
    protocol_handler.register_tool(TOOL_NAME, TOOL_DESCRIPTION, TOOL_INPUT_SCHEMA, tool.execute)
