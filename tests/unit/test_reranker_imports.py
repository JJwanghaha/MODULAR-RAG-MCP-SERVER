"""隔离进程验证后端注册、关闭路径和可选模型依赖的边界。"""

import subprocess
import sys

import pytest

from src.core.settings import REPO_ROOT

pytestmark = pytest.mark.unit


def test_reranker_import_and_disabled_path_need_no_sdk_key_or_network():
    script = '''
import importlib.abc
import os
import socket
import sys
from dataclasses import replace

class NoModelSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sentence_transformers", "google", "openai"}:
            raise ImportError("optional model SDK disabled")

def no_network(*args, **kwargs):
    raise AssertionError("reranker import and disabled path must stay offline")

sys.meta_path.insert(0, NoModelSDK())
socket.socket.connect = no_network
for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
    os.environ.pop(name, None)
from src.core.settings import load_settings
from src.libs.reranker.reranker_factory import RerankerFactory
assert RerankerFactory.list_providers() == ["cross_encoder", "llm"]
settings = load_settings()
items = [{"id": "a", "text": "甲"}, {"id": "b", "text": "乙"}]
for provider in ("llm", "cross_encoder"):
    disabled = replace(settings, rerank=replace(settings.rerank, enabled=False, provider=provider))
    assert RerankerFactory.create(disabled).rerank("问题", items) == items
enabled = replace(settings, rerank=replace(settings.rerank, enabled=True, provider="cross_encoder"))
try:
    RerankerFactory.create(enabled)
except RuntimeError as error:
    assert "sentence-transformers" in str(error)
else:
    raise AssertionError("missing model SDK must prevent real model loading")
print("reranker optional boundary passed")
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "reranker optional boundary passed"
