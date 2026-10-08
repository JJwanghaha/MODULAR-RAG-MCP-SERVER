"""通过公开接口运行真实切分库，验证上游递归切分基线。"""

from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.splitter.splitter_factory import SplitterFactory

pytestmark = pytest.mark.unit


def test_recursive_factory_produces_character_chunks_with_overlap():
    """已讨论的逐字符例子保持原序并重叠一个字符。"""
    splitter = SplitterFactory.create(load_settings(), chunk_size=4, chunk_overlap=1)

    assert splitter.split_text("甲乙丙丁戊己庚辛壬癸") == [
        "甲乙丙丁", "丁戊己庚", "庚辛壬癸",
    ]


@pytest.mark.parametrize("size,overlap,error", [
    (0, 0, "chunk_size"), (-1, 0, "chunk_size"),
    (1.5, 0, "chunk_size"), ("4", 0, "chunk_size"),
    (4, -1, "chunk_overlap"), (4, 1.5, "chunk_overlap"),
    (4, "1", "chunk_overlap"), (4, 4, "less than"), (4, 5, "less than"),
])
def test_factory_reports_invalid_splitter_configuration(size, overlap, error):
    """上游要求正整数大小、非负整数重叠，且重叠严格小于大小。"""
    with pytest.raises(RuntimeError, match=error):
        SplitterFactory.create(load_settings(), chunk_size=size, chunk_overlap=overlap)


def test_direct_splitter_reports_missing_ingestion_settings():
    """直接创建 adapter 时也能定位缺失配置。"""
    from src.libs.splitter.recursive_splitter import RecursiveSplitter

    with pytest.raises(ValueError, match="ingestion"):
        RecursiveSplitter(replace(load_settings(), ingestion=None))


def test_settings_control_the_real_splitter():
    """不传覆盖参数时，实际切分遵循 Settings。"""
    settings = load_settings()
    settings = replace(settings, ingestion=replace(settings.ingestion, chunk_size=4, chunk_overlap=1))
    assert SplitterFactory.create(settings).split_text("甲乙丙丁戊己庚辛壬癸") == [
        "甲乙丙丁", "丁戊己庚", "庚辛壬癸",
    ]


def test_paragraph_merge_does_not_force_exact_overlap():
    """完整段落超过重叠目标时，允许两个输出块没有重叠。"""
    splitter = SplitterFactory.create(load_settings(), chunk_size=8, chunk_overlap=2)
    assert splitter.split_text("甲甲甲\n\n乙乙乙\n\n丙丙丙") == [
        "甲甲甲\n\n乙乙乙", "丙丙丙",
    ]


def test_default_size_counts_characters_not_tokens():
    """默认 1000 字符块与 200 字符目标重叠，最后一块允许更短。"""
    splitter = SplitterFactory.create(load_settings())
    assert splitter.split_text("甲" * 1001) == ["甲" * 1000, "甲" * 201]


def test_custom_separator_is_a_literal_not_a_regular_expression():
    """字面量句号不能将 X 加空格当作句号边界。"""
    splitter = SplitterFactory.create(
        load_settings(), chunk_size=6, chunk_overlap=0, separators=[". ", ""],
    )
    assert splitter.split_text("甲乙丙X 丁戊己") == ["甲乙丙X 丁", "戊己"]


@pytest.mark.parametrize("text", [None, 42, "", " \n "])
def test_invalid_input_is_rejected_before_splitting(text):
    """只接受非空文本字符串，不把非法输入交给库。"""
    splitter = SplitterFactory.create(load_settings())
    with pytest.raises(ValueError):
        splitter.split_text(text)


def test_short_markdown_with_code_and_table_stays_in_one_chunk():
    """短样例能保持原样，不据此声称大代码块或表格受到保护。"""
    text = "# 标题\n\n正文。\n\n```python\nprint(1)\n```\n\n|列|\n|---|\n|值|"
    assert SplitterFactory.create(load_settings()).split_text(text) == [text]


def test_oversized_code_block_exposes_upstream_structure_limit():
    """上游字符切分会把代码围栏和代码正文放进不同块。"""
    splitter = SplitterFactory.create(load_settings(), chunk_size=8, chunk_overlap=0)
    assert splitter.split_text("```\nABCDEFGHIJ\n```") == ["```", "ABCDEFG", "HIJ", "```"]


def test_whitespace_is_trimmed_unless_explicitly_preserved():
    """透传 strip_whitespace 可以改变首尾空白处理。"""
    text = "  # 标题  "
    assert SplitterFactory.create(load_settings()).split_text(text) == ["# 标题"]
    assert SplitterFactory.create(load_settings(), strip_whitespace=False).split_text(text) == [text]


def test_empty_library_output_falls_back_to_original_text():
    """特殊自定义分隔符移除全文时，沿用上游返回原文的回退。"""
    splitter = SplitterFactory.create(
        load_settings(), chunk_size=2, chunk_overlap=0,
        separators=["X"], keep_separator=False,
    )
    assert splitter.split_text("XXX") == ["XXX"]


def test_library_failure_is_reported_with_configuration_context():
    """真实库失败时报告长度与配置，不在错误信息中复制输入正文。"""
    splitter = SplitterFactory.create(load_settings(), separators=[1, ""])
    with pytest.raises(RuntimeError, match="Text length: 2"):
        splitter.split_text("甲乙")


def test_factory_imports_without_optional_library_and_reports_install_hint():
    """缺少可选库不妨碍注册与导入，只阻止创建选中的递归实现。"""
    import subprocess
    import sys
    from src.core.settings import REPO_ROOT

    script = '''
import importlib.abc
import sys

class NoLangChain(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"langchain_text_splitters", "langchain_core"}:
            raise ImportError("optional splitter disabled for this test")

sys.meta_path.insert(0, NoLangChain())
from src.core.settings import load_settings
from src.libs.splitter.splitter_factory import SplitterFactory
assert SplitterFactory.list_providers() == ["recursive"]
try:
    SplitterFactory.create(load_settings())
except RuntimeError as error:
    assert "langchain-text-splitters" in str(error)
    assert "splitters" in str(error)
else:
    raise AssertionError("missing optional library must prevent construction")
print("optional splitter boundary passed")
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "optional splitter boundary passed"
