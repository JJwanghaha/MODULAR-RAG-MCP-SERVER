"""根据配置创建文本或视觉 LLM adapter，分别维护注册表。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM

if TYPE_CHECKING:
    from src.core.settings import Settings


class LLMFactory:
    """维护 provider Registry，并集中创建 LLM adapter。"""

    _PROVIDERS: dict[str, type[BaseLLM]] = {}
    _VISION_PROVIDERS: dict[str, type[BaseVisionLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """注册 provider 名称与 adapter 类的映射。"""
        if not isinstance(provider_class, type) or not issubclass(provider_class, BaseLLM):
            provider_name = getattr(provider_class, "__name__", repr(provider_class))
            raise ValueError(f"Provider class {provider_name} must inherit from BaseLLM")
        cls._PROVIDERS[name.strip().lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseLLM:
        """根据 Settings 中的 provider 创建 adapter。"""
        provider_name = settings.llm.provider.strip().lower()
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(cls.list_providers()) or "none"
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate LLM provider '{provider_name}': {exc}"
            ) from exc

    @classmethod
    def list_providers(cls) -> list[str]:
        """按名称排序返回已注册的 provider。"""
        return sorted(cls._PROVIDERS)

    @classmethod
    def register_vision_provider(cls, name: str, provider_class: type[BaseVisionLLM]) -> None:
        """注册视觉 adapter；不会因此构造对象、读取图片或请求模型。"""
        if not isinstance(provider_class, type) or not issubclass(provider_class, BaseVisionLLM):
            provider_name = getattr(provider_class, "__name__", repr(provider_class))
            raise ValueError(f"Provider class {provider_name} must inherit from BaseVisionLLM")
        cls._VISION_PROVIDERS[name.strip().lower()] = provider_class

    @classmethod
    def create_vision_llm(cls, settings: Settings, **override_kwargs: Any) -> BaseVisionLLM:
        """优先使用视觉 provider，未配置时按上游使用文本 provider 选名。"""
        vision = getattr(settings, "vision_llm", None)
        config = vision if hasattr(vision, "provider") else getattr(settings, "llm", None)
        provider = getattr(config, "provider", None)
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Missing required configuration: settings.vision_llm.provider or settings.llm.provider")
        provider_name = provider.strip().lower()
        provider_class = cls._VISION_PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(cls.list_vision_providers()) or "none"
            raise ValueError(
                f"Unsupported Vision LLM provider: '{provider_name}'. "
                f"Available Vision LLM providers: {available}"
            )
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate Vision LLM provider '{provider_name}': {exc}"
            ) from exc

    @classmethod
    def list_vision_providers(cls) -> list[str]:
        """只返回视觉注册表，不推断文本 adapter 是否具有视觉能力。"""
        return sorted(cls._VISION_PROVIDERS)


# 注册不构造对象，因此导入模块不需要密钥，也不会访问网络。
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.libs.llm.ollama_llm import OllamaLLM
from src.libs.llm.gemini_llm import GeminiLLM

LLMFactory.register_provider("openai", OpenAILLM)
LLMFactory.register_provider("azure", AzureLLM)
LLMFactory.register_provider("deepseek", DeepSeekLLM)
LLMFactory.register_provider("ollama", OllamaLLM)
LLMFactory.register_provider("gemini", GeminiLLM)

# 只注册视觉类；不读取图片、导入图像 SDK 或发出模型请求。
from src.libs.llm.azure_vision_llm import AzureVisionLLM
from src.libs.llm.gemini_vision_llm import GeminiVisionLLM

LLMFactory.register_vision_provider("azure", AzureVisionLLM)
LLMFactory.register_vision_provider("gemini", GeminiVisionLLM)
