"""A3 使用的最小 stderr 日志入口。"""

import logging
import sys


def get_logger(name: str = "modular-rag", log_level: str = "INFO") -> logging.Logger:
    """返回一个不会重复添加 Handler 的标准日志对象。"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    return logger
