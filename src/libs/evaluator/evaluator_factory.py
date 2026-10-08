"""按评估配置选择轻量指标或无操作策略。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    from src.core.settings import Settings


class EvaluatorFactory:
    """B6 提供 custom；模型评估框架在 H 阶段接入。"""

    _PROVIDERS: dict[str, type[BaseEvaluator]] = {"custom": CustomEvaluator}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEvaluator]) -> None:
        """注册评估实现。"""
        if not issubclass(provider_class, BaseEvaluator):
            raise ValueError(f"Provider class {provider_class.__name__} must inherit from BaseEvaluator")
        cls._PROVIDERS[name.strip().lower()] = provider_class

    @classmethod
    def list_providers(cls) -> list[str]:
        """按名称返回注册的评估器。"""
        return sorted(cls._PROVIDERS)

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEvaluator:
        """创建配置指定的评估器。"""
        name = settings.evaluation.provider.strip().lower()
        if not settings.evaluation.enabled or name in {"none", "disabled"}:
            return NoneEvaluator(settings=settings, **override_kwargs)
        provider_class = cls._PROVIDERS.get(name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS)) or "none"
            raise ValueError(f"Unsupported Evaluator provider: '{name}'. Available providers: {available}")
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to instantiate Evaluator provider '{name}': {exc}") from exc
