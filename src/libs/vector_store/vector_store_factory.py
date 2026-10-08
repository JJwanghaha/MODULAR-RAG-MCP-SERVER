"""根据 vector_store.provider 选择具体存储实现。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings


class VectorStoreFactory:
    """B4 只建立注册表，Chroma 实现在 B7.6 注册。"""

    _PROVIDERS: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseVectorStore]) -> None:
        """注册存储实现。"""
        if not issubclass(provider_class, BaseVectorStore):
            raise ValueError(f"Provider class {provider_class.__name__} must inherit from BaseVectorStore")
        cls._PROVIDERS[name.strip().lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseVectorStore:
        """把配置及覆盖参数传给选中的存储类。"""
        name = settings.vector_store.provider.strip().lower()
        provider_class = cls._PROVIDERS.get(name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise ValueError(
                f"Unsupported VectorStore provider: '{name}'. Available providers: {available}"
            )
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to instantiate VectorStore provider '{name}': {exc}") from exc

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回已注册的提供者名称。"""
        return sorted(cls._PROVIDERS)
