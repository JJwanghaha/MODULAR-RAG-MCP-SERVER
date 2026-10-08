"""隔离进程验证 Chroma 注册不依赖 SDK、Key 或数据库初始化。"""

import subprocess
import sys

import pytest

from src.core.settings import REPO_ROOT

pytestmark = pytest.mark.unit


def test_factory_imports_without_chroma_and_reports_optional_dependency():
    """缺 SDK 时仍可列出后端，选中它才返回安装提示。"""
    script = '''
import importlib.abc
import socket
import sys

class NoChroma(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "chromadb":
            raise ImportError("optional database SDK disabled for this test")

def no_network(*args, **kwargs):
    raise AssertionError("factory imports must stay offline")

sys.meta_path.insert(0, NoChroma())
socket.socket.connect = no_network
from src.core.settings import load_settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
assert VectorStoreFactory.list_providers() == ["chroma"]
try:
    VectorStoreFactory.create(load_settings())
except RuntimeError as error:
    assert "chromadb" in str(error)
    assert "vector-stores" in str(error)
else:
    raise AssertionError("missing SDK must prevent database construction")
print("optional Chroma boundary passed")
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "optional Chroma boundary passed"
