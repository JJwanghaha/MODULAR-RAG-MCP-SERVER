"""C1 核心对象的公开构造与序列化契约。"""

import json

import pytest

pytestmark = pytest.mark.unit


def test_document_roundtrip_preserves_text_and_source():
    from src.core.types import Document

    document = Document("doc-1", "# 学习笔记\n\n正文", {"source_path": "note.md", "doc_type": "markdown"})
    encoded = json.dumps(document.to_dict(), ensure_ascii=False)
    restored = Document.from_dict(json.loads(encoded))

    assert restored == document
    assert restored.text == "# 学习笔记\n\n正文"
    assert restored.metadata["source_path"] == "note.md"


def test_document_requires_source_path():
    from src.core.types import Document

    with pytest.raises(ValueError, match="Document metadata must contain 'source_path'"):
        Document("doc-1", "正文")


def test_chunk_record_from_chunk_preserves_vectors_and_independent_metadata():
    from src.core.types import Chunk, ChunkRecord

    chunk = Chunk("chunk-1", "片段", {"source_path": "note.md", "chunk_index": 0},
                  start_offset=3, end_offset=5, source_ref="doc-1")
    record = ChunkRecord.from_chunk(chunk, dense_vector=[0.1, 0.2], sparse_vector={"片段": 1.0})
    record.metadata["title"] = "记录标题"

    assert "title" not in chunk.metadata
    assert record.text == "片段"
    assert record.dense_vector == [0.1, 0.2]
    assert record.sparse_vector == {"片段": 1.0}
    assert Chunk.from_dict(json.loads(json.dumps(chunk.to_dict()))) == chunk
    assert ChunkRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record


@pytest.mark.parametrize("name", ["Document", "Chunk", "ChunkRecord"])
def test_all_core_types_require_source(name):
    from src.core import types

    with pytest.raises(ValueError, match=f"{name} metadata must contain 'source_path'"):
        getattr(types, name)("id", "正文", {})


@pytest.mark.parametrize("name", ["Document", "Chunk", "ChunkRecord"])
def test_to_dict_does_not_share_nested_metadata(name):
    from src.core import types

    value = getattr(types, name)("id", "正文", {"source_path": "a.md", "tags": ["原始"]})
    data = value.to_dict()
    data["metadata"]["tags"].append("修改")

    assert value.metadata["tags"] == ["原始"]


def test_optional_chunk_fields_and_vectors_default_to_none():
    from src.core.types import Chunk, ChunkRecord

    chunk = Chunk("id", "正文", {"source_path": "a.md"})
    record = ChunkRecord.from_chunk(chunk)

    assert (chunk.start_offset, chunk.end_offset, chunk.source_ref) == (None, None, None)
    assert (record.dense_vector, record.sparse_vector) == (None, None)


def test_record_conversion_preserves_upstream_shallow_copy_limit():
    from src.core.types import Chunk, ChunkRecord

    chunk = Chunk("id", "正文", {"source_path": "a.md", "tags": ["原始"]}, start_offset=0, end_offset=2)
    record = ChunkRecord.from_chunk(chunk)

    assert record.metadata is not chunk.metadata
    assert record.metadata["tags"] is chunk.metadata["tags"]
    assert "start_offset" not in record.to_dict()
