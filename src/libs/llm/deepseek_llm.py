"""DeepSeek 使用上游相同的 Chat Completions 协议。"""

from src.libs.llm.openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    """只替换服务地址和认证来源，复用相同消息转换。"""

    PROVIDER = "deepseek"
    KEY_ENV = "DEEPSEEK_API_KEY"
    DEFAULT_BASE_URL = "https://api.deepseek.com"
