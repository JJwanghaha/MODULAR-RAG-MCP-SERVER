"""Markdown 扩展：保留源码结构，输出上游 Document。"""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

from src.core.settings import resolve_path
from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    """图片目录和允许资源范围可指定，不自动创建目录或调用外部系统。"""

    def __init__(self, image_storage_dir="data/images", source_root=None):
        self.image_storage_dir = resolve_path(image_storage_dir)
        self.source_root = Path(source_root).resolve() if source_root is not None else None

    def load(self, file_path: str | Path) -> Document:
        """读取 UTF-8/BOM 文本，规范换行，不重新渲染表格或代码。"""
        path = self._validate_file(file_path)
        if path.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError(f"File is not Markdown: {path}")
        try:
            from markdown_it import MarkdownIt
            from markdown_it.rules_inline.image import image as parse_image
        except ImportError:
            raise ImportError("Markdown parser is not installed; install .[loaders]") from None

        raw = path.read_bytes()
        doc_hash = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        frontmatter = {}
        lines = text.splitlines(keepends=True)
        if lines and lines[0].strip() == "---":
            end = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
            if end is not None:
                try:
                    parsed = yaml.safe_load("".join(lines[1:end]))
                    if isinstance(parsed, dict):
                        frontmatter = json.loads(json.dumps(parsed, default=str, ensure_ascii=False, allow_nan=False))
                        text = "".join(lines[end + 1:])
                except (yaml.YAMLError, TypeError, ValueError):
                    pass
        parser = MarkdownIt("commonmark").enable("table")

        def image_rule(state, silent):
            start = state.pos
            accepted = parse_image(state, silent)
            if accepted and not silent:
                state.tokens[-1].meta["source_span"] = (start, state.pos)
                state.tokens[-1].meta["source_markup"] = state.src[start:state.pos]
            return accepted

        parser.inline.ruler.at("image", image_rule)
        tokens = parser.parse(text)
        title = path.stem
        for index, token in enumerate(tokens):
            if token.type == "heading_open" and token.tag == "h1":
                title = tokens[index + 1].content
                break
        head_title = frontmatter.get("title")
        if isinstance(head_title, str) and head_title.strip():
            title = head_title.strip()
        text, images, unprocessed = self._process_images(text, tokens, path, doc_hash)
        return Document(f"doc_{doc_hash[:16]}", text, {
            "source_path": str(path), "doc_type": "markdown", "doc_hash": doc_hash,
            "title": title, "frontmatter": frontmatter, "images": images, "unprocessed_images": unprocessed,
        })

    def _process_images(self, text, tokens, path, doc_hash):
        """只处理解析器识别的图片，不用全文正则替换代码或特殊语法。"""
        lines = text.splitlines(keepends=True)
        starts = [0]
        for line in lines:
            starts.append(starts[-1] + len(line))
        cursor = 0
        spans = []
        unprocessed = []
        for token in tokens:
            if token.type != "inline" or not token.content or token.map is None:
                continue
            begin, end = starts[token.map[0]], starts[token.map[1]]
            scan = max(begin, cursor)
            mapping = []
            for part in token.content.splitlines(keepends=True):
                core = part[:-1] if part.endswith("\n") else part
                found = text.find(core, scan, end)
                if found < 0:
                    mapping = []
                    break
                mapping.extend(range(found, found + len(core)))
                scan = found + len(core)
                if part.endswith("\n"):
                    newline = text.find("\n", scan, end)
                    if newline < 0:
                        mapping = []
                        break
                    mapping.append(newline)
                    scan = newline + 1
            if mapping:
                cursor = scan
            for image in token.children or []:
                if image.type != "image":
                    continue
                a, b = image.meta["source_span"]
                if b > len(mapping) or not mapping:
                    unprocessed.append({"original_ref": image.attrGet("src"), "alt": image.content,
                                        "reason": "unsupported_source_span"})
                    continue
                left, right = mapping[a], mapping[b - 1] + 1
                if text[left:right] != image.meta["source_markup"]:
                    unprocessed.append({"original_ref": image.attrGet("src"), "alt": image.content,
                                        "reason": "unsupported_source_span"})
                    continue
                spans.append((left, right, image))
        spans.sort(key=lambda value: value[0])
        result, images = [], []
        previous, length = 0, 0
        root = self.source_root or path.parent
        for index, (left, right, image) in enumerate(spans, 1):
            before = text[previous:left]
            result.append(before)
            length += len(before)
            src = image.attrGet("src") or ""
            alt = image.content
            image_id = f"{doc_hash}_md_{index}"
            info, reason = self._copy_image(src, root, path.parent, doc_hash, image_id)
            if info is None:
                markup = text[left:right]
                unprocessed.append({"original_ref": src, "alt": alt, "reason": reason,
                                    "text_offset": length, "text_length": len(markup)})
                result.append(markup)
                length += len(markup)
            else:
                placeholder = f"[IMAGE: {image_id}]"
                prefix = f"{alt} " if alt else ""
                result.append(prefix + placeholder)
                images.append({**info, "id": image_id, "alt": alt, "original_ref": src,
                               "text_offset": length + len(prefix), "text_length": len(placeholder)})
                length += len(prefix) + len(placeholder)
            previous = right
        result.append(text[previous:])
        return "".join(result), images, unprocessed

    def _copy_image(self, src, root, parent, doc_hash, image_id):
        """只接入限定范围内的 PNG/JPEG/WebP 文件；不请求任何 URL。"""
        try:
            parsed = urlsplit(src)
        except ValueError:
            return None, "invalid_target"
        if parsed.scheme in {"http", "https"} or parsed.netloc:
            return None, "remote_not_fetched"
        if parsed.scheme:
            return None, "unsupported_scheme"
        try:
            local = Path(unquote(parsed.path))
            candidate = (local if local.is_absolute() else parent / local).resolve()
        except (ValueError, OSError):
            return None, "invalid_target"
        if not candidate.is_relative_to(root):
            return None, "outside_source_root"
        if not candidate.is_file():
            return None, "not_found"
        try:
            from PIL import Image
        except ImportError:
            return None, "missing_image_dependency"
        try:
            with Image.open(candidate) as image:
                format = image.format
                image.verify()
            formats = {"PNG": ("png", "image/png"), "JPEG": ("jpg", "image/jpeg"), "WEBP": ("webp", "image/webp")}
            if format not in formats:
                return None, "unsupported_image_format"
            extension, mime = formats[format]
            directory = self.image_storage_dir / doc_hash
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{image_id}.{extension}"
            shutil.copyfile(candidate, destination)
            return {"path": str(destination), "mime_type": mime}, None
        except (OSError, ValueError, shutil.Error):
            return None, "image_read_or_copy_failed"
