"""隔离进程验证注册后端不需要密钥、可选 SDK 或网络。"""

import os
import subprocess
import sys

import pytest

from src.core.settings import REPO_ROOT

pytestmark = pytest.mark.unit


def test_builtin_factories_import_without_keys_or_optional_sdks():
    """只安装基础依赖也能导入全部后端，并创建无密钥的 Ollama 对象。"""
    script = '''
import importlib.abc
import socket
import sys
from dataclasses import replace

class NoModelSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"openai", "google"}:
            raise ImportError("optional model SDK disabled for this test")

def no_network(*args, **kwargs):
    raise AssertionError("model imports must not access network")

sys.meta_path.insert(0, NoModelSDK())
socket.socket.connect = no_network
from src.core.settings import load_settings
from src.libs.llm.llm_factory import LLMFactory
from src.libs.embedding.embedding_factory import EmbeddingFactory

assert LLMFactory.list_providers() == ["azure", "deepseek", "gemini", "ollama", "openai"]
assert EmbeddingFactory.list_providers() == ["azure", "gemini", "ollama", "openai"]
settings = load_settings()
LLMFactory.create(replace(settings, llm=replace(settings.llm, provider="ollama")))
EmbeddingFactory.create(replace(settings, embedding=replace(settings.embedding, provider="ollama")))
print("offline imports passed")
'''
    environment = {key: value for key, value in os.environ.items() if key not in {
        "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    }}
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline imports passed"
