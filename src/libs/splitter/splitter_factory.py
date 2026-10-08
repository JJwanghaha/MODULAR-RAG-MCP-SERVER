"""按 ingestion.splitter 配置选择切分策略。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from src.core.settings import Settings


class SplitterFactory:
    """注册和创建切分器；内置递归字符切分策略。"""

    _PROVIDERS: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseSplitter]) -> None:
        """注册策略名称与具体类。"""
        if not issubclass(provider_class, BaseSplitter):
            raise ValueError(f"Provider class {provider_class.__name__} must inherit from BaseSplitter")
        cls._PROVIDERS[name.strip().lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseSplitter:
        """创建配置指定的策略，并传递参数覆盖值。"""
        try:
            provider_name = settings.ingestion.splitter.strip().lower()
        except AttributeError as exc:
            raise ValueError("Missing required configuration: settings.ingestion.splitter") from exc
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise ValueError(
                f"Unsupported Splitter provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate Splitter provider '{provider_name}': {exc}"
            ) from exc

    @classmethod
    def list_providers(cls) -> list[str]:
        """返回按名称排序的已注册策略。"""
        return sorted(cls._PROVIDERS)


# 只注册类，创建递归切分器时才导入可选依赖。
from src.libs.splitter.recursive_splitter import RecursiveSplitter

SplitterFactory.register_provider("recursive", RecursiveSplitter)
