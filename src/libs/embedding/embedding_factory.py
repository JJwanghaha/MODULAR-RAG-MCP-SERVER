"""根据配置创建 Embedding adapter。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings


class EmbeddingFactory:
    """维护 provider Registry，并集中创建 Embedding adapter。"""

    _PROVIDERS: dict[str, type[BaseEmbedding]] = {}

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: type[BaseEmbedding],
    ) -> None:
        """注册 provider 名称与 adapter 类的映射。"""
        if not isinstance(provider_class, type) or not issubclass(provider_class, BaseEmbedding):
            provider_name = getattr(provider_class, "__name__", repr(provider_class))
            raise ValueError(
                f"Provider class {provider_name} must inherit from BaseEmbedding"
            )
        cls._PROVIDERS[name.strip().lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEmbedding:
        """根据 Settings 中的 provider 创建 adapter。"""
        provider_name = settings.embedding.provider.strip().lower()
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise ValueError(
                f"Unsupported Embedding provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate Embedding provider '{provider_name}': {exc}"
            ) from exc

    @classmethod
    def list_providers(cls) -> list[str]:
        """按名称排序返回已注册的 provider。"""
        return sorted(cls._PROVIDERS)
