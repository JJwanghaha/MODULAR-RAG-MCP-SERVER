"""可选 Loader 依赖和离线边界的隔离进程验收。"""

import os
import subprocess
import sys

import pytest

from src.core.settings import REPO_ROOT

pytestmark = pytest.mark.unit


def test_loader_imports_and_construction_need_no_optional_sdks(tmp_path):
    script = '''
import importlib.abc
import socket
import sys
from pathlib import Path

class NoOptionalSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"markdown_it", "markitdown", "fitz", "pymupdf", "PIL", "google", "openai"}:
            raise ImportError("optional SDK disabled")

def no_network(*args, **kwargs):
    raise AssertionError("Loader must not access network")

sys.meta_path.insert(0, NoOptionalSDK())
socket.socket.connect = no_network
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
root = Path(sys.argv[1])
for filename, loader in [("note.md", MarkdownLoader(image_storage_dir=root / "managed")),
                         ("note.pdf", PdfLoader(image_storage_dir=root / "managed"))]:
    path = root / filename
    path.write_bytes(b"fixture")
    try:
        loader.load(path)
    except ImportError as error:
        assert "install .[loaders]" in str(error)
    else:
        raise AssertionError("missing parser must prevent loading")
assert not (root / "managed").exists()
assert not (root / "data").exists()
print("loader optional boundary passed")
'''
    environment = {key: value for key, value in os.environ.items() if key not in {
        "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "AZURE_OPENAI_API_KEY",
    }}
    environment["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "loader optional boundary passed"


def test_base_loader_is_abstract():
    from src.libs.loader.base_loader import BaseLoader

    with pytest.raises(TypeError, match="abstract"):
        BaseLoader()
