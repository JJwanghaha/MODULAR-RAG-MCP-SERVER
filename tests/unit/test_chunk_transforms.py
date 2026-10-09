"""C5–C7 的公开 transform 契约，只替代外部模型后端。"""

from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.core.types import Chunk
from src.libs.llm.base_llm import BaseLLM, ChatResponse

pytestmark = pytest.mark.unit


class ModelStub(BaseLLM):
    """固定模型响应或错误，不替换清理和解析逻辑。"""

    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.prompts = []

    def chat(self, messages, trace=None, **kwargs):
        self.prompts.append(messages[0].content)
        if self.error:
            raise self.error
        return ChatResponse(self.content, "stub-model")


def chunk(text, id="c1", **metadata):
    return Chunk(id, text, {"source_path": "note.md", **metadata}, source_ref="doc1")


def test_c5_rule_cleaning_preserves_complete_code_and_image_marker_without_llm():
    from src.ingestion.transform.chunk_refiner import ChunkRefiner

    original = chunk(" # Title  \n\n\n<div>正文   内容</div><!-- noise -->\n\n```python\n  print('  a')\n```\n[IMAGE: a]")
    model = ModelStub(error=AssertionError("disabled model must not be called"))
    result = ChunkRefiner(load_settings(), llm=model).transform([original])

    assert result[0].text == "# Title\n\n正文 内容\n\n```python\n  print('  a')\n```\n[IMAGE: a]"
    assert result[0].metadata["refined_by"] == "rule"
    assert result[0].id == original.id
    assert result[0].source_ref == "doc1"
    assert original.text.startswith(" # Title  ")
    assert model.prompts == []


def test_c5_enabled_llm_uses_prompt_and_falls_back_when_backend_fails(tmp_path):
    from src.ingestion.transform.chunk_refiner import ChunkRefiner

    settings = load_settings()
    settings = replace(settings, ingestion=replace(settings.ingestion, chunk_refiner={"use_llm": True}))
    prompt = tmp_path / "refine.txt"
    prompt.write_text("清理：{text}", encoding="utf-8")
    original = chunk("正文   内容 [IMAGE: a]")
    successful = ModelStub("清理结果 [IMAGE: a]")
    result = ChunkRefiner(settings, llm=successful, prompt_path=prompt).transform([original])
    assert result[0].text == "清理结果 [IMAGE: a]"
    assert result[0].metadata["refined_by"] == "llm"
    assert successful.prompts == ["清理：正文 内容 [IMAGE: a]"]

    failed = ChunkRefiner(settings, llm=ModelStub(error=RuntimeError("backend failure")), prompt_path=prompt).transform([original])
    assert failed[0].text == "正文 内容 [IMAGE: a]"
    assert failed[0].metadata["refined_by"] == "rule"


def test_c6_rule_metadata_does_not_replace_body():
    from src.ingestion.transform.metadata_enricher import MetadataEnricher

    original = chunk("# 架构\nPython uses camelCase and **检索**.", title="文档标题")
    result = MetadataEnricher(load_settings()).transform([original])[0]

    assert result.text == original.text
    assert result.metadata["title"] == "架构"
    assert result.metadata["summary"] == original.text
    assert "检索" in result.metadata["tags"]
    assert result.metadata["enriched_by"] == "rule"
    assert original.metadata["title"] == "文档标题"


def test_c6_model_fields_and_failure_fallback_use_real_parser():
    from src.ingestion.transform.metadata_enricher import MetadataEnricher

    settings = load_settings()
    settings = replace(settings, ingestion=replace(settings.ingestion, metadata_enricher={"use_llm": True}))
    original = chunk("# 原文标题\n正文")
    model = ModelStub("Title: 模型标题\nSummary: 模型摘要\nTags: RAG, 检索\n")
    result = MetadataEnricher(settings, llm=model).transform([original])[0]

    assert result.metadata["title"] == "模型标题"
    assert result.metadata["summary"] == "模型摘要"
    assert result.metadata["tags"] == ["RAG", "检索"]
    assert result.metadata["enriched_by"] == "llm"
    assert result.text == original.text
    assert original.text in model.prompts[0]
    failed = MetadataEnricher(settings, llm=ModelStub(error=RuntimeError("backend failure"))).transform([original])[0]
    assert failed.metadata["enriched_by"] == "rule"
    assert failed.metadata["enrich_fallback_reason"] == "llm_failed"


def test_c7_unique_referenced_image_is_captioned_once_and_written_to_each_chunk(tmp_path):
    from src.core.settings import VisionLLMSettings
    from src.libs.llm.base_vision_llm import BaseVisionLLM
    from src.ingestion.transform.image_captioner import ImageCaptioner

    class VisionStub(BaseVisionLLM):
        def __init__(self):
            self.paths = []

        def chat_with_image(self, text, image, messages=None, trace=None, **kwargs):
            self.paths.append(image.path)
            return ChatResponse("模拟图像内容", "stub-vision")

    path = tmp_path / "image.png"
    path.write_bytes(b"test image carrier, not real image decoding")
    settings = replace(load_settings(), vision_llm=VisionLLMSettings(True, "gemini", "stub", 2048))
    image = {"id": "a", "path": str(path)}
    unrelated = {"id": "unused", "path": str(tmp_path / "unused.png")}
    values = [chunk("[IMAGE: a]", "c1", images=[image, unrelated]), chunk("再次 [IMAGE: a]", "c2", images=[image])]
    model = VisionStub()
    result = ImageCaptioner(settings, llm=model).transform(values)

    assert model.paths == [str(path)]
    assert [value.id for value in result] == ["c1", "c2"]
    assert all("(Description: 模拟图像内容)" in value.text for value in result)
    assert all(value.metadata["image_captions"] == [{"id": "a", "caption": "模拟图像内容"}] for value in result)
    assert result[0] is values[0]


@pytest.mark.parametrize("mode", ["disabled", "failed", "missing"])
def test_c7_unavailable_vision_keeps_original_placeholder(tmp_path, mode):
    from src.core.settings import VisionLLMSettings
    from src.libs.llm.base_vision_llm import BaseVisionLLM
    from src.ingestion.transform.image_captioner import ImageCaptioner

    class UnavailableVision(BaseVisionLLM):
        def __init__(self):
            self.paths = []

        def chat_with_image(self, text, image, messages=None, trace=None, **kwargs):
            self.paths.append(image.path)
            raise RuntimeError("simulated vision failure")

    path = tmp_path / "image.png"
    if mode != "missing":
        path.write_bytes(b"fixture")
    settings = load_settings()
    if mode != "disabled":
        settings = replace(settings, vision_llm=VisionLLMSettings(True, "gemini", "stub", 2048))
    value = chunk("[IMAGE: a]", images=[{"id": "a", "path": str(path)}])
    model = UnavailableVision()
    result = ImageCaptioner(settings, llm=model).transform([value])

    assert result[0] is value
    assert result[0].text == "[IMAGE: a]"
    assert "image_captions" not in result[0].metadata
    assert "has_unprocessed_images" not in result[0].metadata
    assert len(model.paths) == (1 if mode == "failed" else 0)


@pytest.mark.parametrize("mode", ["empty", "missing_prompt", "missing_placeholder"])
def test_c5_optional_model_or_prompt_problem_keeps_rule_result(tmp_path, mode):
    from src.ingestion.transform.chunk_refiner import ChunkRefiner

    settings = load_settings()
    settings = replace(settings, ingestion=replace(settings.ingestion, chunk_refiner={"use_llm": True}))
    prompt = tmp_path / "refine.txt"
    if mode != "missing_prompt":
        prompt.write_text("{text}" if mode == "empty" else "没有占位符", encoding="utf-8")
    result = ChunkRefiner(settings, llm=ModelStub(""), prompt_path=prompt).transform([chunk("正文   内容")])
    assert result[0].text == "正文 内容"
    assert result[0].metadata["refined_by"] == "rule"


def test_bad_chunk_does_not_block_other_chunks_and_empty_batch_is_valid():
    from src.ingestion.transform.chunk_refiner import ChunkRefiner
    from src.ingestion.transform.metadata_enricher import MetadataEnricher
    from src.ingestion.transform.image_captioner import ImageCaptioner

    bad = chunk(42, "bad")
    refiner = ChunkRefiner(load_settings())
    results = refiner.transform([bad, chunk("正文   内容", "good")])
    assert results[0] is bad
    assert results[1].text == "正文 内容"
    enriched = MetadataEnricher(load_settings()).transform([chunk(None, "bad"), chunk("# 好块", "good")])
    assert [value.id for value in enriched] == ["bad", "good"]
    assert enriched[0].metadata["enriched_by"] == "error"
    assert enriched[1].metadata["title"] == "好块"
    assert refiner.transform([]) == []
    assert MetadataEnricher(load_settings()).transform([]) == []
    assert ImageCaptioner(load_settings()).transform([]) == []


def test_c6_keeps_upstream_lenient_response_parsing():
    from src.ingestion.transform.metadata_enricher import MetadataEnricher

    settings = load_settings()
    settings = replace(settings, ingestion=replace(settings.ingestion, metadata_enricher={"use_llm": True}))
    result = MetadataEnricher(settings, llm=ModelStub("非标准响应")).transform([chunk("正文")])[0]
    assert result.metadata["title"] == "Untitled"
    assert result.metadata["summary"] == "非标准响应"
    assert result.metadata["tags"] == []
    assert result.metadata["enriched_by"] == "llm"


def test_rule_refinement_keeps_upstream_markdown_indentation_limit():
    from src.ingestion.transform.chunk_refiner import ChunkRefiner

    value = chunk("- 一级\n    - 二级\n\n结束  \n下一行")
    result = ChunkRefiner(load_settings()).transform([value])[0]
    assert result.text == "- 一级\n - 二级\n\n结束\n下一行"


def test_c5_c6_c7_public_chain_adds_caption_after_rule_enrichment(tmp_path):
    from src.core.settings import VisionLLMSettings
    from src.libs.llm.base_vision_llm import BaseVisionLLM
    from src.ingestion.transform.chunk_refiner import ChunkRefiner
    from src.ingestion.transform.metadata_enricher import MetadataEnricher
    from src.ingestion.transform.image_captioner import ImageCaptioner

    class VisionStub(BaseVisionLLM):
        def chat_with_image(self, text, image, messages=None, trace=None, **kwargs):
            return ChatResponse("测试图像说明", "stub")

    path = tmp_path / "image.png"
    path.write_bytes(b"fixture")
    settings = replace(load_settings(), vision_llm=VisionLLMSettings(True, "gemini", "stub", 2048))
    original = chunk("# 标题\n\n正文   内容\n[IMAGE: a]", images=[{"id": "a", "path": str(path)}])
    refined = ChunkRefiner(settings).transform([original])
    enriched = MetadataEnricher(settings).transform(refined)
    result = ImageCaptioner(settings, llm=VisionStub()).transform(enriched)[0]
    assert result.metadata["refined_by"] == "rule"
    assert result.metadata["enriched_by"] == "rule"
    assert result.metadata["title"] == "标题"
    assert "正文 内容" in result.text
    assert "测试图像说明" in result.text
    assert original.text.endswith("[IMAGE: a]")


def test_ingestion_transform_settings_loads_flags_and_rejects_string_boolean(tmp_path):
    import yaml
    from src.core.settings import DEFAULT_SETTINGS_PATH, SettingsError

    data = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    data["ingestion"]["chunk_refiner"] = {"use_llm": False}
    data["ingestion"]["metadata_enricher"] = {"use_llm": True}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = load_settings(path)
    assert settings.ingestion.chunk_refiner == {"use_llm": False}
    assert settings.ingestion.metadata_enricher == {"use_llm": True}
    data["ingestion"]["chunk_refiner"]["use_llm"] = "false"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(SettingsError, match="ingestion.chunk_refiner.use_llm"):
        load_settings(path)


def test_default_transforms_work_without_model_sdks_keys_or_network(tmp_path):
    import os
    import subprocess
    import sys
    from src.core.settings import REPO_ROOT

    script = '''
import importlib.abc
import socket
import sys

class NoModelSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"google", "openai"}:
            raise ImportError("model SDK disabled")

def no_network(*args, **kwargs):
    raise AssertionError("default transforms must remain offline")

sys.meta_path.insert(0, NoModelSDK())
socket.socket.connect = no_network
from src.core.settings import load_settings
from src.core.types import Chunk
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
settings = load_settings()
result = ChunkRefiner(settings).transform([Chunk("id", "# Title\\ntext   body", {"source_path": "a.md"})])
result = MetadataEnricher(settings).transform(result)
assert result[0].text == "# Title\\ntext body"
assert ImageCaptioner(settings).transform(result) is result
print("default transforms offline passed")
'''
    environment = {key: value for key, value in os.environ.items() if key not in {
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY",
    }}
    environment["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "default transforms offline passed"
