"""Embedding interface 与 Factory 的公开行为测试。"""

from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


pytestmark = pytest.mark.unit


class FakeEmbedding(BaseEmbedding):
    """返回确定向量、不访问网络的测试 adapter。"""

    def __init__(self, settings=None, dimension: int = 3, **kwargs):
        self.settings = settings
        self.dimension = dimension
        self.kwargs = kwargs

    def embed(self, texts, trace=None, **kwargs):
        """为每段文本返回固定规则生成的向量。"""
        self.validate_texts(texts)
        return [
            [float(text_index + value_index) for value_index in range(self.dimension)]
            for text_index in range(len(texts))
        ]

    def get_dimension(self) -> int:
        """返回当前 adapter 的向量维度。"""
        return self.dimension


def _isolated_factory() -> type[EmbeddingFactory]:
    """为每个测试创建独立 Registry。"""

    class IsolatedEmbeddingFactory(EmbeddingFactory):
        _PROVIDERS = {}

    return IsolatedEmbeddingFactory


def _settings_with_provider(provider: str):
    """复用默认配置，只替换当前测试关注的 Embedding provider。"""
    settings = load_settings()
    return replace(settings, embedding=replace(settings.embedding, provider=provider))


def test_fake_embedding_returns_one_vector_per_text() -> None:
    """批量输入与向量输出保持一一对应。"""
    embedding = FakeEmbedding(dimension=2)

    vectors = embedding.embed(["RAG", "Embedding", "MCP"])

    assert vectors == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    assert embedding.get_dimension() == 2


def test_validate_texts_rejects_empty_list() -> None:
    """Embedding 调用不能缺少文本。"""
    embedding = FakeEmbedding()

    with pytest.raises(ValueError, match="Texts list cannot be empty"):
        embedding.validate_texts([])


def test_validate_texts_rejects_non_string_value() -> None:
    """每个输入项都必须是字符串。"""
    embedding = FakeEmbedding()

    with pytest.raises(ValueError, match="Text at index 1 is not a string"):
        embedding.validate_texts(["有效文本", 123])  # type: ignore[list-item]


def test_validate_texts_rejects_blank_value() -> None:
    """空字符串和纯空白字符串不能进入向量模型。"""
    embedding = FakeEmbedding()

    with pytest.raises(ValueError, match="Text at index 1 is empty or whitespace-only"):
        embedding.validate_texts(["有效文本", "   "])


def test_fake_embedding_validates_input() -> None:
    """测试 adapter 在编码前执行公共输入校验。"""
    embedding = FakeEmbedding()

    with pytest.raises(ValueError, match="Texts list cannot be empty"):
        embedding.embed([])


def test_factory_registers_and_creates_provider_with_overrides() -> None:
    """Factory 根据 Settings 创建 adapter，并传递构造参数。"""
    factory = _isolated_factory()
    factory.register_provider("fake", FakeEmbedding)
    settings = _settings_with_provider("fake")

    embedding = factory.create(settings, dimension=4)

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.settings == settings
    assert embedding.get_dimension() == 4
    assert factory.list_providers() == ["fake"]


def test_factory_normalizes_provider_name() -> None:
    """注册和配置中的 provider 名称大小写不敏感。"""
    factory = _isolated_factory()
    factory.register_provider("FAKE", FakeEmbedding)
    settings = _settings_with_provider("fake")

    embedding = factory.create(settings)

    assert isinstance(embedding, FakeEmbedding)
    assert factory.list_providers() == ["fake"]


def test_factory_rejects_non_embedding_class() -> None:
    """Registry 只接受实现 BaseEmbedding interface 的类。"""
    factory = _isolated_factory()

    class NotAnEmbedding:
        pass

    with pytest.raises(ValueError, match="must inherit from BaseEmbedding"):
        factory.register_provider("invalid", NotAnEmbedding)  # type: ignore[arg-type]


def test_factory_reports_unknown_provider() -> None:
    """未知 provider 错误包含请求值和可用列表。"""
    factory = _isolated_factory()
    factory.register_provider("fake", FakeEmbedding)
    settings = _settings_with_provider("unknown")

    with pytest.raises(ValueError) as exc_info:
        factory.create(settings)

    message = str(exc_info.value)
    assert "Unsupported Embedding provider: 'unknown'" in message
    assert "Available providers: fake" in message


def test_factory_reports_provider_construction_failure() -> None:
    """Adapter 构造失败时，错误包含 provider 名和原始原因。"""
    factory = _isolated_factory()

    class BrokenEmbedding(BaseEmbedding):
        def __init__(self, settings=None, **kwargs):
            raise RuntimeError("boom")

        def embed(self, texts, trace=None, **kwargs):
            return [[0.0] for _ in texts]

    factory.register_provider("broken", BrokenEmbedding)
    settings = _settings_with_provider("broken")

    with pytest.raises(RuntimeError) as exc_info:
        factory.create(settings)

    message = str(exc_info.value)
    assert "Failed to instantiate Embedding provider 'broken'" in message
    assert "boom" in message


def test_factory_lists_providers_in_stable_order() -> None:
    """Provider 列表不受注册顺序影响。"""
    factory = _isolated_factory()
    factory.register_provider("zebra", FakeEmbedding)
    factory.register_provider("alpha", FakeEmbedding)
    factory.register_provider("beta", FakeEmbedding)

    assert factory.list_providers() == ["alpha", "beta", "zebra"]


def test_get_dimension_requires_adapter_implementation() -> None:
    """未声明维度的 adapter 不能静默返回错误值。"""

    class IncompleteEmbedding(BaseEmbedding):
        def embed(self, texts, trace=None, **kwargs):
            return [[0.0] for _ in texts]

    embedding = IncompleteEmbedding()

    with pytest.raises(NotImplementedError, match="must implement get_dimension"):
        embedding.get_dimension()
