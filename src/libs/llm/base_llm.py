"""所有文本 LLM adapter 共同遵循的 interface。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    """一条角色明确的对话消息。"""

    role: str
    content: str


@dataclass
class ChatResponse:
    """屏蔽不同模型厂商返回格式的统一响应。"""

    content: str
    model: str
    usage: dict[str, int] | None = None
    raw_response: Any | None = None


class BaseLLM(ABC):
    """文本 LLM adapter 的抽象基类。"""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """根据消息列表生成统一响应。"""
        raise NotImplementedError

    def validate_messages(self, messages: list[Message]) -> None:
        """验证所有 adapter 共享的消息前置条件。"""
        if not messages:
            raise ValueError("Messages list cannot be empty")

        valid_roles = {"system", "user", "assistant"}
        for index, message in enumerate(messages):
            if not isinstance(message, Message):
                raise ValueError(f"Message at index {index} is not a Message instance")
            if message.role not in valid_roles:
                raise ValueError(
                    f"Message at index {index} has invalid role '{message.role}'"
                )
            if not message.content or not message.content.strip():
                raise ValueError(f"Message at index {index} has empty content")
