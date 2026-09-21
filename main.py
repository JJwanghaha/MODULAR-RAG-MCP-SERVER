"""Modular RAG MCP Server 的启动入口。"""

import sys

from src.core.settings import SettingsError, load_settings
from src.observability.logger import get_logger


def main() -> int:
    """加载配置；MCP Server 将在 E 阶段接入。"""
    try:
        settings = load_settings()
    except SettingsError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    logger = get_logger(__name__, settings.observability.log_level)
    logger.info("Settings loaded successfully.")
    logger.info("MCP Server will be implemented in Phase E.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
