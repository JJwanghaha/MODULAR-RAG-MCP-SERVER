"""按重排配置选择策略；真实重排模型留给 B7。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker

if TYPE_CHECKING:
    from src.core.settings import Settings


class RerankerFactory:
    """关闭时选择 NoneReranker，开启时查找注册策略。"""

    _PROVIDERS: dict[str, type[BaseReranker]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseReranker]) -> None:
        """注册重排策略。"""
        if not issubclass(provider_class, BaseReranker):
            raise ValueError(f"Provider class {provider_class.__name__} must inherit from BaseReranker")
        cls._PROVIDERS[name.strip().lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseReranker:
        """创建符合配置的重排对象。"""
        name = settings.rerank.provider.strip().lower()
        if not settings.rerank.enabled or name == "none":
            return NoneReranker(settings=settings, **override_kwargs)
        provider_class = cls._PROVIDERS.get(name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise ValueError(f"Unsupported Reranker provider: '{name}'. Available providers: {available}")
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to instantiate Reranker provider '{name}': {exc}") from exc

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回注册的策略名，无操作回退由 create 单独处理。"""
        return sorted(cls._PROVIDERS)
