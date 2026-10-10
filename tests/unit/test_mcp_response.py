"""E3/E6：引用、证据响应和图片内容的公开边界测试。"""

import base64
import io
from pathlib import Path

from PIL import Image, PngImagePlugin
import pytest

from src.core.response.citation_generator import CitationGenerator
from src.core.response.multimodal_assembler import MultimodalAssembler
from src.core.response.response_builder import ResponseBuilder
from src.core.types import RetrievalResult
from src.ingestion.storage.image_storage import ImageStorage


pytestmark = pytest.mark.unit


def test_citations_and_markdown_response_preserve_source_score_and_evidence():
    """引用保留来源哈希和原始分数，正文使用编号标记且不伪装成百分比。"""
    result = RetrievalResult(
        "chunk-17", 1.375, "第一行\n第二行",
        {
            "source_path": "/private/docs/guide.md", "page_num": 4, "title": "指南",
            "source_ref": "doc_1234567890abcdef", "doc_hash": "a" * 64,
            "internal_only": "不应出现在引用中",
        },
    )

    citation = CitationGenerator().generate([result])[0]
    response = ResponseBuilder(enable_multimodal=False).build([result], "检索问题", "notes")
    mcp_result = response.to_mcp_result()

    assert citation.index == 1
    assert citation.source == "/private/docs/guide.md"
    assert citation.score == 1.375
    assert citation.page == 4
    assert citation.metadata == {
        "title": "指南", "source_ref": "doc_1234567890abcdef", "doc_hash": "a" * 64,
    }
    assert "### [1] 结果 1" in response.content
    assert "第一行 第二行" in response.content
    assert "分数：1.375000" in response.content
    assert "百分比" not in response.content
    assert mcp_result.structuredContent["citations"][0]["metadata"]["source_ref"] == "doc_1234567890abcdef"
    assert mcp_result.structuredContent["isEmpty"] is False
    assert mcp_result.isError is False


def test_empty_response_is_a_normal_mcp_result():
    """无命中返回清楚的空态，不生成空引用或业务错误。"""
    response = ResponseBuilder(enable_multimodal=False).build([], "没有命中的问题", "empty")

    result = response.to_mcp_result()

    assert response.is_empty is True
    assert response.citations == []
    assert "未找到相关结果" in response.content
    assert result.isError is False
    assert result.structuredContent["isEmpty"] is True
    assert result.structuredContent["citations"] == []


def _save_image(path: Path, image: Image.Image, image_format: str, **options) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format=image_format, **options)
    return path.read_bytes()


def test_response_returns_verified_image_formats_and_deduplicates_by_full_content(tmp_path):
    """PNG/JPEG/WebP 返回真实像素；重复字节去重而头部相同的不同 PNG 均保留。"""
    images_root = tmp_path / "images"
    index_path = tmp_path / "db" / "image_index.db"
    storage = ImageStorage(str(index_path), str(images_root))

    first_png = Image.new("RGB", (32, 32), (20, 30, 40))
    second_png = first_png.copy()
    second_png.putpixel((31, 31), (200, 210, 220))
    first_bytes = _save_image(images_root / "first.png", first_png, "PNG", compress_level=0)
    second_bytes = _save_image(images_root / "second.png", second_png, "PNG", compress_level=0)
    alias_bytes = _save_image(images_root / "first-copy.png", first_png, "PNG", compress_level=0)
    _save_image(images_root / "photo.jpg", Image.new("RGB", (8, 8), (90, 40, 12)), "JPEG")
    _save_image(images_root / "photo.webp", Image.new("RGB", (8, 8), (12, 90, 40)), "WEBP")
    assert first_bytes[:100] == second_bytes[:100]
    assert first_bytes != second_bytes
    assert alias_bytes == first_bytes

    paths = {
        "png-first": images_root / "first.png", "png-second": images_root / "second.png",
        "png-alias": images_root / "first-copy.png", "jpeg": images_root / "photo.jpg",
        "webp": images_root / "photo.webp",
    }
    for image_id, path in paths.items():
        storage.register_image(image_id, path, "notes")

    assembler = MultimodalAssembler(images_root=images_root, image_index_path=index_path)
    evidence = RetrievalResult(
        "chunk-with-image", 0.625,
        "正文仍可阅读。[IMAGE: png-first] [IMAGE: png-second] [IMAGE: jpeg] [IMAGE: webp]",
        {"image_refs": "png-first,png-second,png-alias,jpeg,webp", "images": "不是列表，不能反序列化"},
    )
    response = ResponseBuilder(multimodal_assembler=assembler).build([evidence], "图片问题", "notes")
    result = response.to_mcp_result()
    image_blocks = [item for item in result.content if item.type == "image"]

    assert "正文仍可阅读。[IMAGE: png-first]" in response.content
    assert result.structuredContent["has_images"] is True
    assert result.structuredContent["image_count"] == 4
    assert len(image_blocks) == 4
    assert {item.mimeType for item in image_blocks} == {"image/png", "image/jpeg", "image/webp"}
    decoded = [base64.b64decode(item.data) for item in image_blocks]
    assert len(set(decoded)) == 4
    png_decoded = [data for data in decoded if data.startswith(b"\x89PNG\r\n\x1a\n")]
    assert len(png_decoded) == 2
    assert png_decoded[0] != png_decoded[1]
    decoded_images = []
    for data in decoded:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            last_pixel = (image.size[0] - 1, image.size[1] - 1)
            decoded_images.append((image.format, image.size, image.getpixel((0, 0)), image.getpixel(last_pixel)))
    assert {item[0] for item in decoded_images} == {"PNG", "JPEG", "WEBP"}
    assert {item[1] for item in decoded_images} == {(32, 32), (8, 8)}
    png_pixels = [item[3] for item in decoded_images if item[0] == "PNG"]
    assert set(png_pixels) == {(20, 30, 40), (200, 210, 220)}


def test_multimodal_assembler_skips_missing_invalid_oversized_outside_symlink_and_other_collection(tmp_path):
    """越界、失效、超限和跨集合引用不返回任意文件内容。"""
    images_root = tmp_path / "managed-images"
    index_path = tmp_path / "db" / "image_index.db"
    storage = ImageStorage(str(index_path), str(images_root))
    outside_path = tmp_path / "outside.png"
    _save_image(outside_path, Image.new("RGB", (2, 2), "red"), "PNG")
    storage.register_image("outside", outside_path, "notes")

    invalid_path = images_root / "notes" / "invalid.png"
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_bytes(b"not an image")
    storage.register_image("invalid", invalid_path, "notes")

    oversized_path = images_root / "notes" / "oversized.png"
    oversized = PngImagePlugin.PngInfo()
    oversized.add_text("payload", "x" * (2 * 1024 * 1024 + 128))
    Image.new("RGB", (1, 1), "blue").save(oversized_path, format="PNG", pnginfo=oversized)
    assert oversized_path.stat().st_size > 2 * 1024 * 1024
    storage.register_image("oversized", oversized_path, "notes")

    gif_path = images_root / "notes" / "unsupported.gif"
    _save_image(gif_path, Image.new("RGB", (2, 2), "purple"), "GIF")
    storage.register_image("unsupported-gif", gif_path, "notes")
    svg_path = images_root / "notes" / "unsupported.svg"
    svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"></svg>', encoding="utf-8")
    storage.register_image("unsupported-svg", svg_path, "notes")

    other_collection_path = images_root / "other" / "private.png"
    _save_image(other_collection_path, Image.new("RGB", (2, 2), "green"), "PNG")
    storage.register_image("private", other_collection_path, "other")

    symlink_path = images_root / "notes" / "escape.png"
    symlink_path.symlink_to(outside_path)
    evidence = RetrievalResult(
        "unsafe-references", 0.5, "保留这段正文。",
        {"image_refs": "outside,invalid,oversized,unsupported-gif,unsupported-svg,private,missing,remote", "images": [
            {"id": "direct-outside", "path": str(outside_path)},
            {"id": "symlink", "path": str(symlink_path)},
            {"id": "missing", "path": str(images_root / "missing.png")},
            {"id": "remote", "path": "https://example.invalid/image.png"},
        ]},
    )
    assembler = MultimodalAssembler(images_root=images_root, image_index_path=index_path, max_images_per_result=10)

    response = ResponseBuilder(multimodal_assembler=assembler).build([evidence], "安全边界", "notes")

    assert "保留这段正文。" in response.content
    assert response.image_contents == []
    assert response.has_images is False


def test_multimodal_assembler_caps_total_images_at_five(tmp_path):
    """多块引用累计时最多返回五张有效图片。"""
    images_root = tmp_path / "images"
    index_path = tmp_path / "db" / "image_index.db"
    storage = ImageStorage(str(index_path), str(images_root))
    identifiers = []
    for index in range(6):
        image_id = f"image-{index}"
        path = images_root / f"{image_id}.png"
        _save_image(path, Image.new("RGB", (2, 2), (index * 30, index * 20, index * 10)), "PNG")
        storage.register_image(image_id, path, "notes")
        identifiers.append(image_id)
    evidence = [
        RetrievalResult("first-image-group", 0.4, "前三张关联图片。", {"image_refs": ",".join(identifiers[:3])}),
        RetrievalResult("second-image-group", 0.3, "后三张关联图片。", {"image_refs": ",".join(identifiers[3:])}),
    ]

    assembler = MultimodalAssembler(
        images_root=images_root, image_index_path=index_path, max_images_per_result=5, max_images_total=5,
    )

    assert assembler.count_images(evidence) == 6
    assert len(assembler.assemble(evidence, "notes")) == 5
