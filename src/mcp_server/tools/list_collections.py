"""E4：列出已有 Chroma 集合，count 表示 Chunk 记录数而非文章数。"""

import asyncio
from dataclasses import dataclass
from typing import Any

from src.mcp_server.tools._storage import configured_storage, existing_chroma_client

TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = "列出可用知识库集合；可返回每个集合的 Chunk 记录数，不调用模型。"
TOOL_INPUT_SCHEMA = {"type": "object", "properties": {"include_stats": {"type": "boolean", "default": True}},
                     "additionalProperties": False}


@dataclass
class CollectionInfo:
    name: str
    count: int | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"name": self.name}
        if self.count is not None:
            result["count"] = self.count
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


class ListCollectionsTool:
    def __init__(self, settings=None, *, data_dir=None):
        self.settings, self.data_root = configured_storage(settings, data_dir)

    def list_collections(self, include_stats: bool = True) -> list[CollectionInfo]:
        if not isinstance(include_stats, bool):
            raise ValueError("include_stats must be a boolean")
        try:
            with existing_chroma_client(self.settings) as client:
                results = []
                for item in client.list_collections():
                    name = item.name if hasattr(item, "name") else str(item)
                    collection = client.get_collection(name, embedding_function=None)
                    metadata = {key: value for key, value in (collection.metadata or {}).items()
                                if not key.startswith(("_", "hnsw:"))}
                    results.append(CollectionInfo(name, collection.count() if include_stats else None, metadata))
                return sorted(results, key=lambda item: item.name)
        except FileNotFoundError:
            return []

    def format_response(self, collections: list[CollectionInfo]) -> str:
        if not collections:
            return "暂无知识库集合，请先摄取文档。"
        return "## 知识库集合\n\n" + "\n".join(
            f"- `{item.name}`" + (f"：{item.count} 个 Chunk 记录" if item.count is not None else "") for item in collections)

    async def execute(self, include_stats: bool = True):
        from mcp import types
        collections = await asyncio.to_thread(self.list_collections, include_stats)
        return types.CallToolResult(content=[types.TextContent(type="text", text=self.format_response(collections))],
                                    structuredContent={"collections": [item.to_dict() for item in collections]}, isError=False)


def register_tool(protocol_handler, settings=None, *, data_dir=None) -> None:
    tool = ListCollectionsTool(settings, data_dir=data_dir)
    protocol_handler.register_tool(TOOL_NAME, TOOL_DESCRIPTION, TOOL_INPUT_SCHEMA, tool.execute)
