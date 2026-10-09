"""C4 公开入口验证，实际运行已有切分器，不替换内部算法。"""

from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.core.types import Chunk, Document

pytestmark = pytest.mark.unit


def settings_for(size=1000, overlap=200):
    """只修改切分参数，其他配置保持默认。"""
    settings = load_settings()
    return replace(settings, ingestion=replace(settings.ingestion, chunk_size=size, chunk_overlap=overlap))


def test_chunker_uses_upstream_stable_id_and_wraps_real_splitter_result():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_123", "Hello world", {"source_path": "hello.md", "title": "示例"})
    chunks = DocumentChunker(load_settings()).split_document(document)

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == "Hello world"
    assert chunks[0].id == "doc_123_0000_64ec88ca"
    assert chunks[0].metadata["source_path"] == "hello.md"
    assert chunks[0].metadata["title"] == "示例"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["source_ref"] == "doc_123"


def test_each_chunk_carries_only_its_referenced_images():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    images = [
        {"id": "img_a", "path": "a.png", "page": 2, "text_offset": 4},
        {"id": "img_b", "path": "b.png", "page": 3, "text_offset": 20},
    ]
    document = Document("doc_images", "前文\n\n[IMAGE: img_a]\n\n[IMAGE: img_b]\n\n结尾", {
        "source_path": "images.pdf", "images": images,
    })
    chunks = DocumentChunker(settings_for(16, 0)).split_document(document)

    assert [chunk.text for chunk in chunks] == ["前文", "[IMAGE: img_a]", "[IMAGE: img_b]", "结尾"]
    assert [chunk.metadata["image_refs"] for chunk in chunks] == [[], ["img_a"], ["img_b"], []]
    assert "images" not in chunks[0].metadata
    assert "images" not in chunks[3].metadata
    assert chunks[1].metadata["images"] == [images[0]]
    assert chunks[2].metadata["images"] == [images[1]]
    assert chunks[1].metadata["page_num"] == 2
    assert chunks[2].metadata["page_num"] == 3
    assert document.metadata["images"] == images


@pytest.mark.parametrize("text", ["", " \n "])
def test_blank_document_reports_document_id(text):
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_empty", text, {"source_path": "empty.md"})
    with pytest.raises(ValueError, match="Document doc_empty has no text content to split"):
        DocumentChunker(load_settings()).split_document(document)


def test_unknown_image_reference_does_not_invent_image_path():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_unknown", "[IMAGE: missing]", {"source_path": "note.md", "images": None})
    chunk = DocumentChunker(load_settings()).split_document(document)[0]

    assert chunk.metadata["image_refs"] == ["missing"]
    assert "images" not in chunk.metadata
    assert "page_num" not in chunk.metadata


def test_settings_control_actual_fragments_and_overlap():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_overlap", "甲乙丙丁戊己庚辛壬癸", {"source_path": "note.md"})
    chunks = DocumentChunker(settings_for(4, 1)).split_document(document)

    assert [chunk.text for chunk in chunks] == ["甲乙丙丁", "丁戊己庚", "庚辛壬癸"]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert len({chunk.id for chunk in chunks}) == 3


def test_repeat_split_is_deterministic_and_same_content_positions_have_distinct_ids():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_repeat", "abc\n\nabc\n\nabc", {"source_path": "repeat.md"})
    chunker = DocumentChunker(settings_for(6, 0))
    chunks = chunker.split_document(document)

    assert chunks == chunker.split_document(document)
    assert [chunk.text for chunk in chunks] == ["abc", "abc", "abc"]
    assert [chunk.id for chunk in chunks] == [
        "doc_repeat_0000_ba7816bf", "doc_repeat_0001_ba7816bf", "doc_repeat_0002_ba7816bf",
    ]


def test_document_identity_and_changed_text_affect_ids():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    chunker = DocumentChunker(load_settings())
    first = chunker.split_document(Document("doc_a", "abc", {"source_path": "a.md"}))[0]
    changed_document = chunker.split_document(Document("doc_b", "abc", {"source_path": "a.md"}))[0]
    changed_text = chunker.split_document(Document("doc_a", "abcd", {"source_path": "a.md"}))[0]

    assert first.id == "doc_a_0000_ba7816bf"
    assert changed_document.id == "doc_b_0000_ba7816bf"
    assert changed_text.id != first.id


def test_reserved_chunk_metadata_is_rebuilt_without_changing_document():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_parent", "abc", {
        "source_path": "a.md", "chunk_index": 99, "source_ref": "old", "image_refs": ["old"],
    })
    before = document.to_dict()
    chunk = DocumentChunker(load_settings()).split_document(document)[0]

    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["source_ref"] == "doc_parent"
    assert chunk.metadata["image_refs"] == []
    assert document.to_dict() == before


def test_nested_metadata_and_image_records_keep_upstream_shallow_copy_semantics():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    image = {"id": "a", "path": "a.png", "page": 1, "text_offset": 100}
    document = Document("doc_shared", "[IMAGE: a]", {
        "source_path": "a.pdf", "tags": ["RAG"], "images": [image],
    })
    chunk = DocumentChunker(load_settings()).split_document(document)[0]

    assert chunk.metadata is not document.metadata
    assert chunk.metadata["tags"] is document.metadata["tags"]
    assert chunk.metadata["images"] is not document.metadata["images"]
    assert chunk.metadata["images"][0] is image
    assert chunk.metadata["images"][0]["text_offset"] == 100


def test_independent_chunk_offset_and_source_ref_fields_remain_unfilled():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_refs", "abc", {"source_path": "a.md"})
    chunk = DocumentChunker(load_settings()).split_document(document)[0]

    assert (chunk.start_offset, chunk.end_offset, chunk.source_ref) == (None, None, None)
    assert chunk.metadata["source_ref"] == "doc_refs"


def test_first_image_supplies_page_hint_not_full_chunk_page_range():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    images = [{"id": "b", "path": "b.png", "page": 3}, {"id": "a", "path": "a.png", "page": 2}]
    document = Document("doc_pages", "[IMAGE: a]\n[IMAGE: b]", {"source_path": "a.pdf", "images": images})
    chunk = DocumentChunker(load_settings()).split_document(document)[0]

    assert chunk.metadata["image_refs"] == ["a", "b"]
    assert chunk.metadata["images"] == [images[1], images[0]]
    assert chunk.metadata["page_num"] == 2


def test_markdown_image_without_page_does_not_get_a_fabricated_page():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_md", "[IMAGE: a]", {
        "source_path": "a.md", "images": [{"id": "a", "path": "a.png", "alt": "图"}],
    })
    chunk = DocumentChunker(load_settings()).split_document(document)[0]

    assert chunk.metadata["page_num"] is None


def test_repeated_placeholder_refs_are_not_deduplicated():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    image = {"id": "a", "path": "a.png"}
    document = Document("doc_repeat_img", "[IMAGE: a] [IMAGE: a]", {"source_path": "a.md", "images": [image]})
    chunk = DocumentChunker(load_settings()).split_document(document)[0]

    assert chunk.metadata["image_refs"] == ["a", "a"]
    assert chunk.metadata["images"] == [image, image]


def test_whitespace_in_placeholder_id_is_trimmed():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_whitespace", "[IMAGE:  a  ]", {
        "source_path": "a.md", "images": [{"id": "a", "path": "a.png"}],
    })
    chunk = DocumentChunker(load_settings()).split_document(document)[0]
    assert chunk.metadata["image_refs"] == ["a"]


def test_cut_placeholder_exposes_existing_splitter_limit():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_cut", "[IMAGE: image_a]", {
        "source_path": "a.md", "images": [{"id": "image_a", "path": "a.png"}],
    })
    chunks = DocumentChunker(settings_for(6, 0)).split_document(document)

    assert len(chunks) > 1
    assert all(chunk.metadata["image_refs"] == [] for chunk in chunks)
    assert all("images" not in chunk.metadata for chunk in chunks)


def test_literal_placeholder_inside_code_is_still_seen_by_upstream_regex():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    document = Document("doc_code", "```\n[IMAGE: a]\n```", {
        "source_path": "a.md", "images": [{"id": "a", "path": "a.png"}],
    })
    chunk = DocumentChunker(load_settings()).split_document(document)[0]

    assert chunk.metadata["image_refs"] == ["a"]


def test_metadata_only_change_preserves_content_id():
    from src.ingestion.chunking.document_chunker import DocumentChunker

    chunker = DocumentChunker(load_settings())
    old = chunker.split_document(Document("doc_meta", "abc", {"source_path": "a.md", "title": "旧"}))[0]
    new = chunker.split_document(Document("doc_meta", "abc", {"source_path": "a.md", "title": "新"}))[0]

    assert old.id == new.id
    assert old.metadata["title"] != new.metadata["title"]


@pytest.mark.parametrize("case", ["missing", "unknown"])
def test_chunker_uses_factory_configuration_errors(case):
    from src.ingestion.chunking.document_chunker import DocumentChunker

    settings = load_settings()
    if case == "missing":
        settings = replace(settings, ingestion=None)
        match = "settings.ingestion.splitter"
    else:
        settings = replace(settings, ingestion=replace(settings.ingestion, splitter="unknown"))
        match = "Unsupported Splitter provider: 'unknown'"
    with pytest.raises(ValueError, match=match):
        DocumentChunker(settings)


def test_markdown_loader_output_can_be_split_with_image_subset(tmp_path):
    from PIL import Image
    from src.libs.loader.markdown_loader import MarkdownLoader
    from src.ingestion.chunking.document_chunker import DocumentChunker

    with Image.new("RGB", (10, 10), "blue") as image:
        image.save(tmp_path / "flow.png")
    path = tmp_path / "note.md"
    path.write_text("# 标题\n\n" + "第一段内容。" * 80 + "\n\n![流程图](flow.png)\n\n结尾", encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)
    original = document.to_dict()
    chunks = DocumentChunker(settings_for(180, 0)).split_document(document)
    with_images = [chunk for chunk in chunks if chunk.metadata["image_refs"]]

    assert len(chunks) > 1
    assert len(with_images) == 1
    assert with_images[0].metadata["images"] == document.metadata["images"]
    assert with_images[0].metadata["page_num"] is None
    assert all(chunk.metadata["source_ref"] == document.id for chunk in chunks)
    assert document.to_dict() == original


def test_import_needs_no_model_keys_and_optional_splitter_is_loaded_on_creation(tmp_path):
    import os
    import subprocess
    import sys
    from src.core.settings import REPO_ROOT

    script = '''
import importlib.abc
import socket
import sys
from pathlib import Path

class NoOptionalSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"langchain_text_splitters", "google", "openai"}:
            raise ImportError("optional SDK disabled")

def no_network(*args, **kwargs):
    raise AssertionError("C4 must not access external systems")

sys.meta_path.insert(0, NoOptionalSDK())
socket.socket.connect = no_network
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.core.settings import load_settings
try:
    DocumentChunker(load_settings())
except RuntimeError as error:
    assert "install .[splitters]" in str(error)
else:
    raise AssertionError("missing splitter SDK must prevent creation")
assert not Path("data").exists()
print("chunker optional boundary passed")
'''
    environment = {key: value for key, value in os.environ.items() if key not in {
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY",
    }}
    environment["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "chunker optional boundary passed"
