"""C3 真实 PDF 解析；只生成非敏感临时样例，不调用模型。"""

import pytest

pytestmark = pytest.mark.unit


def test_pdf_loader_extracts_real_text_and_source(tmp_path, monkeypatch):
    import socket
    from reportlab.pdfgen import canvas
    from src.libs.loader.pdf_loader import PdfLoader

    path = tmp_path / "sample.pdf"
    document = canvas.Canvas(str(path))
    document.drawString(72, 740, "RAG sample document")
    document.drawString(72, 710, "Retrieval uses document evidence.")
    document.save()

    def no_network(*args, **kwargs):
        raise AssertionError("PDF parsing must remain local")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    parsed = PdfLoader(image_storage_dir=tmp_path / "images").load(path)

    assert "RAG sample document" in parsed.text
    assert "Retrieval uses document evidence." in parsed.text
    assert parsed.metadata["source_path"] == str(path.resolve())
    assert parsed.metadata["doc_type"] == "pdf"
    assert len(parsed.metadata["doc_hash"]) == 64


def test_pdf_images_are_saved_and_placeholders_match_offsets(tmp_path):
    import io
    from pathlib import Path
    from PIL import Image
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from src.libs.loader.pdf_loader import PdfLoader

    output = io.BytesIO()
    with Image.new("RGB", (40, 20), "green") as image:
        image.save(output, format="PNG")
    path = tmp_path / "with-image.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 740, "PDF with an embedded image")
    pdf.drawImage(ImageReader(io.BytesIO(output.getvalue())), 72, 650, width=80, height=40)
    pdf.save()
    loader = PdfLoader(image_storage_dir=tmp_path / "managed")
    document = loader.load(path)

    assert len(document.metadata["images"]) == 1
    info = document.metadata["images"][0]
    assert info["page"] == 1
    assert Path(info["path"]).is_file()
    assert document.text[info["text_offset"]:info["text_offset"] + info["text_length"]] == f"[IMAGE: {info['id']}]"
    assert info["position"]["width"] == 40
    assert info["position"]["height"] == 20
    assert loader.load(path) == document


@pytest.fixture
def image_pdf(tmp_path):
    import io
    from PIL import Image
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    data = io.BytesIO()
    with Image.new("RGB", (40, 20), "red") as image:
        image.save(data, format="PNG")
    path = tmp_path / "pages.pdf"
    pdf = canvas.Canvas(str(path))
    for index in range(2):
        pdf.drawString(72, 740, f"Page {index + 1} evidence")
        pdf.drawImage(ImageReader(io.BytesIO(data.getvalue())), 72, 650, width=80, height=40)
        pdf.showPage()
    pdf.save()
    return path


def test_pdf_can_disable_images_without_writing_assets(tmp_path, image_pdf):
    from src.libs.loader.pdf_loader import PdfLoader

    output = tmp_path / "managed"
    document = PdfLoader(extract_images=False, image_storage_dir=output).load(image_pdf)
    assert "Page 1 evidence" in document.text
    assert document.metadata.get("images", []) == []
    assert not output.exists()


def test_pdf_image_page_metadata_and_append_only_position_are_explicit(tmp_path, image_pdf):
    from src.libs.loader.pdf_loader import PdfLoader

    document = PdfLoader(image_storage_dir=tmp_path / "managed").load(image_pdf)
    assert [item["page"] for item in document.metadata["images"]] == [1, 2]
    assert document.text.index("[IMAGE:") > document.text.index("Page 2 evidence")
    assert len({item["id"] for item in document.metadata["images"]}) == 2


def test_optional_pdf_image_extraction_failure_keeps_text(tmp_path, image_pdf, monkeypatch):
    import pymupdf as fitz
    from src.libs.loader.pdf_loader import PdfLoader

    def unavailable(self, xref):
        raise RuntimeError("simulated image failure")
    monkeypatch.setattr(fitz.Document, "extract_image", unavailable)
    document = PdfLoader(image_storage_dir=tmp_path / "managed").load(image_pdf)
    assert "Page 1 evidence" in document.text
    assert document.metadata.get("images", []) == []


def test_pdf_asset_write_failure_keeps_text(tmp_path, image_pdf):
    from src.libs.loader.pdf_loader import PdfLoader

    output = tmp_path / "blocked"
    output.write_bytes(b"not-a-directory")
    document = PdfLoader(image_storage_dir=output).load(image_pdf)
    assert "Page 1 evidence" in document.text
    assert document.metadata.get("images", []) == []


@pytest.mark.parametrize("kind,error", [("missing", FileNotFoundError), ("directory", ValueError), ("extension", ValueError), ("invalid", RuntimeError)])
def test_pdf_invalid_source_has_readable_failure(tmp_path, kind, error):
    from src.libs.loader.pdf_loader import PdfLoader

    path = tmp_path / "bad.pdf"
    if kind == "directory":
        path = tmp_path
    elif kind == "extension":
        path = tmp_path / "bad.md"
        path.write_bytes(b"text")
    elif kind == "invalid":
        path.write_bytes(b"not-a-pdf")
    with pytest.raises(error):
        PdfLoader(image_storage_dir=tmp_path / "managed").load(path)
