"""C3 Markdown Loader 的公开解析输出，文件均为临时样例。"""

import pytest
from pathlib import Path

pytestmark = pytest.mark.unit


def test_markdown_preserves_structure_and_returns_document(tmp_path):
    from src.core.types import Document
    from src.libs.loader.markdown_loader import MarkdownLoader

    text = "# 知识笔记\n\n## 参数\n\n| 参数 | 值 |\n|---|---|\n| top_k | 5 |\n\n```python\nprint('hello')\n```\n"
    path = tmp_path / "note.md"
    path.write_text(text, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "images").load(path)

    assert isinstance(document, Document)
    assert document.text == text
    assert document.metadata["source_path"] == str(path.resolve())
    assert document.metadata["doc_type"] == "markdown"
    assert document.metadata["title"] == "知识笔记"
    assert document.metadata["images"] == []


def test_valid_frontmatter_is_metadata_not_body_and_can_be_json_encoded(tmp_path):
    import json
    from src.libs.loader.markdown_loader import MarkdownLoader

    path = tmp_path / "note.md"
    path.write_text("---\ntitle: 头部标题\ntags: [RAG, Agent]\ndate: 2026-10-09\nsource_path: forbidden-override\n---\n# 正文标题\n\n内容\n", encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "images").load(path)

    assert document.text == "# 正文标题\n\n内容\n"
    assert document.metadata["title"] == "头部标题"
    assert document.metadata["frontmatter"]["date"] == "2026-10-09"
    assert document.metadata["source_path"] == str(path.resolve())
    json.dumps(document.to_dict(), ensure_ascii=False)


def test_local_image_is_copied_with_alt_and_exact_placeholder_offset(tmp_path):
    from PIL import Image
    from src.libs.loader.markdown_loader import MarkdownLoader

    assets = tmp_path / "assets"
    assets.mkdir()
    source = assets / "flow.png"
    with Image.new("RGB", (20, 10), "red") as image:
        image.save(source)
    path = tmp_path / "note.md"
    original = "# 标题\n\n![流程图](assets/flow.png)\n"
    path.write_text(original, encoding="utf-8")
    output = tmp_path / "managed"

    document = MarkdownLoader(image_storage_dir=output).load(path)
    assert len(document.metadata["images"]) == 1
    info = document.metadata["images"][0]
    placeholder = f"[IMAGE: {info['id']}]"
    assert document.text[info["text_offset"]:info["text_offset"] + info["text_length"]] == placeholder
    assert "流程图" in document.text
    assert info["alt"] == "流程图"
    assert info["original_ref"] == "assets/flow.png"
    assert info["mime_type"] == "image/png"
    assert source.read_bytes() == Path(info["path"]).read_bytes()
    assert path.read_text(encoding="utf-8") == original


@pytest.fixture
def picture(tmp_path):
    from PIL import Image

    path = tmp_path / "flow.png"
    with Image.new("RGB", (20, 10), "blue") as image:
        image.save(path)
    return path


@pytest.mark.parametrize("syntax", [
    "![图](flow.png)",
    "![图][diagram]\n\n[diagram]: flow.png",
    "![diagram][]\n\n[diagram]: flow.png",
    "![diagram]\n\n[diagram]: flow.png",
    "> ![图](flow.png)",
    "- ![图](flow.png)",
    "**![图](flow.png)**",
    "| 图 | 示例 |\n|---|---|\n| ![图](flow.png) | 内容 |",
])
def test_standard_images_resolve_without_rendering_source(tmp_path, picture, syntax):
    from src.libs.loader.markdown_loader import MarkdownLoader

    path = tmp_path / "note.md"
    path.write_text(syntax, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)

    assert len(document.metadata["images"]) == 1
    info = document.metadata["images"][0]
    assert document.text[info["text_offset"]:info["text_offset"] + info["text_length"]] == f"[IMAGE: {info['id']}]"
    assert Path(info["path"]).read_bytes() == picture.read_bytes()


def test_code_and_escaped_images_are_not_replaced(tmp_path, picture):
    from src.libs.loader.markdown_loader import MarkdownLoader

    literal = "![图](flow.png)"
    prefix = f"```markdown\n{literal}\n```\n\n`{literal}`\n\n\\{literal}\n\n"
    path = tmp_path / "note.md"
    path.write_text(prefix + literal, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)

    assert document.text.startswith(prefix)
    assert len(document.metadata["images"]) == 1
    assert document.metadata["images"][0]["text_offset"] > len(prefix)


def test_repeated_images_have_distinct_offsets_and_stable_ids(tmp_path, picture):
    from src.libs.loader.markdown_loader import MarkdownLoader

    path = tmp_path / "note.md"
    path.write_text("![图](flow.png)\n\n![图](flow.png)", encoding="utf-8")
    loader = MarkdownLoader(image_storage_dir=tmp_path / "managed")
    first, second = loader.load(path), loader.load(path)

    assert first == second
    assert len({item["id"] for item in first.metadata["images"]}) == 2
    for info in first.metadata["images"]:
        assert first.text[info["text_offset"]:info["text_offset"] + info["text_length"]] == f"[IMAGE: {info['id']}]"


def test_images_in_table_do_not_replace_code_in_previous_cell(tmp_path, picture):
    from src.libs.loader.markdown_loader import MarkdownLoader

    path = tmp_path / "note.md"
    path.write_text("| 代码 | 图 |\n|---|---|\n| `![图](flow.png)` | ![图](flow.png) |", encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)

    assert "`![图](flow.png)`" in document.text
    assert len(document.metadata["images"]) == 1


@pytest.mark.parametrize("header", [
    "---\nnot: [valid\n---\n", "---\n普通说明\n---\n", "---\n未闭合头部\n",
])
def test_invalid_or_non_mapping_frontmatter_is_preserved(tmp_path, header):
    from src.libs.loader.markdown_loader import MarkdownLoader

    text = header + "# 标题\n正文"
    path = tmp_path / "note.md"
    path.write_text(text, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)

    assert document.text == text
    assert document.metadata["frontmatter"] == {}


@pytest.mark.parametrize("target,reason", [
    ("https://example.invalid/image.png", "remote_not_fetched"),
    ("missing.png", "not_found"),
    ("data:image/png;base64,abcdef", "unsupported_scheme"),
])
def test_unprocessed_images_preserve_markup_and_do_not_fetch(tmp_path, monkeypatch, target, reason):
    import socket
    from src.libs.loader.markdown_loader import MarkdownLoader

    def no_network(*args, **kwargs):
        raise AssertionError("Markdown must not download resources")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    path = tmp_path / "note.md"
    text = f"# 标题\n\n![图]({target})"
    path.write_text(text, encoding="utf-8")
    output = tmp_path / "managed"
    document = MarkdownLoader(image_storage_dir=output).load(path)

    assert document.text == text
    assert document.metadata["images"] == []
    assert document.metadata["unprocessed_images"][0]["reason"] == reason
    assert not output.exists()


def test_resource_scope_blocks_outside_and_symlink_escape(tmp_path, picture):
    from src.libs.loader.markdown_loader import MarkdownLoader

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "link.png").symlink_to(picture)
    path = docs / "note.md"
    text = "![外部](../flow.png)\n![链接](link.png)"
    path.write_text(text, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed", source_root=docs).load(path)

    assert document.text == text
    assert [item["reason"] for item in document.metadata["unprocessed_images"]] == ["outside_source_root", "outside_source_root"]


def test_explicit_resource_root_allows_sibling_assets(tmp_path, picture):
    from src.libs.loader.markdown_loader import MarkdownLoader

    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "note.md"
    path.write_text("![图](../flow.png)", encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed", source_root=tmp_path).load(path)

    assert len(document.metadata["images"]) == 1


def test_bom_crlf_and_markdown_extension_are_supported(tmp_path):
    from src.libs.loader.markdown_loader import MarkdownLoader

    path = tmp_path / "note.MARKDOWN"
    path.write_bytes(b"\xef\xbb\xbf# Title\r\n\r\ntext\r\n")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)

    assert document.text == "# Title\n\ntext\n"
    assert document.metadata["title"] == "Title"


def test_plain_document_uses_filename_and_empty_input_is_not_fabricated(tmp_path):
    from src.libs.loader.markdown_loader import MarkdownLoader

    path = tmp_path / "empty.md"
    path.write_bytes(b"")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)

    assert document.text == ""
    assert document.metadata["title"] == "empty"
    assert document.metadata["doc_hash"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert document.id == "doc_e3b0c44298fc1c14"


def test_html_wiki_and_mermaid_are_preserved_not_executed(tmp_path):
    from src.libs.loader.markdown_loader import MarkdownLoader

    text = '<img src="remote.png">\n\n![[flow.png]]\n\n![file](file:///outside.png)\n\n```mermaid\ngraph TD; A-->B;\n```\n'
    path = tmp_path / "special.md"
    path.write_text(text, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)

    assert document.text == text
    assert document.metadata["images"] == []


@pytest.mark.parametrize("kind,error", [("missing", FileNotFoundError), ("directory", ValueError), ("extension", ValueError), ("encoding", UnicodeDecodeError)])
def test_invalid_markdown_source_reports_failure(tmp_path, kind, error):
    from src.libs.loader.markdown_loader import MarkdownLoader

    path = tmp_path / "bad.md"
    if kind == "directory":
        path = tmp_path
    elif kind == "extension":
        path = tmp_path / "bad.docx"
        path.write_bytes(b"text")
    elif kind == "encoding":
        path.write_bytes(b"\xff\xfe\xff")
    with pytest.raises(error):
        MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)


def test_malformed_optional_image_target_does_not_break_body(tmp_path):
    from src.libs.loader.markdown_loader import MarkdownLoader

    text = "# 正文\n\n![图](http://[broken)"
    path = tmp_path / "note.md"
    path.write_text(text, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)

    assert document.text == text
    assert document.metadata["images"] == []
    assert document.metadata["unprocessed_images"][0]["reason"] == "remote_not_fetched"


def test_frontmatter_removal_and_multiple_images_use_final_document_offsets(tmp_path, picture):
    from src.libs.loader.markdown_loader import MarkdownLoader

    path = tmp_path / "note.md"
    path.write_text("---\ntitle: 测试\n---\n# 正文\n\n![本地](flow.png)\n\n![远程](https://example.invalid/x.png)", encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)
    for info in document.metadata["images"]:
        assert document.text[info["text_offset"]:info["text_offset"] + info["text_length"]] == f"[IMAGE: {info['id']}]"
    unresolved = document.metadata["unprocessed_images"][0]
    assert document.text[unresolved["text_offset"]:unresolved["text_offset"] + unresolved["text_length"]] == "![远程](https://example.invalid/x.png)"


@pytest.mark.parametrize("name,link", [
    ("图 流程.png", "<图 流程.png>"),
    ("flow(test).png", "flow(test).png"),
    ("with space.png", "with%20space.png"),
])
def test_common_local_image_targets_are_resolved(tmp_path, picture, name, link):
    from src.libs.loader.markdown_loader import MarkdownLoader

    target = tmp_path / name
    target.write_bytes(picture.read_bytes())
    path = tmp_path / "note.md"
    path.write_text(f"![图]({link})", encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)
    assert len(document.metadata["images"]) == 1


def test_unknown_inline_source_mapping_preserves_markup(tmp_path, picture):
    from src.libs.loader.markdown_loader import MarkdownLoader

    (tmp_path / "flow|x.png").write_bytes(picture.read_bytes())
    text = "| 图片 |\n|---|\n| ![图](flow\\|x.png) |"
    path = tmp_path / "note.md"
    path.write_text(text, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)
    assert document.text == text
    assert document.metadata["unprocessed_images"][0]["reason"] == "unsupported_source_span"


@pytest.mark.parametrize("suffix", [".txt", ".svg"])
def test_invalid_or_unsupported_local_image_is_preserved(tmp_path, suffix):
    from src.libs.loader.markdown_loader import MarkdownLoader

    target = tmp_path / f"image{suffix}"
    target.write_text("not-a-raster-image", encoding="utf-8")
    text = f"![图]({target.name})"
    path = tmp_path / "note.md"
    path.write_text(text, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)
    assert document.text == text
    assert document.metadata["unprocessed_images"][0]["reason"] == "image_read_or_copy_failed"


def test_copy_failure_does_not_remove_image_reference(tmp_path, picture):
    from src.libs.loader.markdown_loader import MarkdownLoader

    output = tmp_path / "blocked"
    output.write_bytes(b"not-a-directory")
    path = tmp_path / "note.md"
    text = "![图](flow.png)"
    path.write_text(text, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=output).load(path)
    assert document.text == text
    assert document.metadata["unprocessed_images"][0]["reason"] == "image_read_or_copy_failed"


def test_invalid_decoded_resource_path_does_not_break_document(tmp_path):
    from src.libs.loader.markdown_loader import MarkdownLoader

    text = "# 正文\n\n![图](bad%00.png)"
    path = tmp_path / "note.md"
    path.write_text(text, encoding="utf-8")
    document = MarkdownLoader(image_storage_dir=tmp_path / "managed").load(path)
    assert document.text == text
    assert document.metadata["unprocessed_images"][0]["reason"] == "invalid_target"
