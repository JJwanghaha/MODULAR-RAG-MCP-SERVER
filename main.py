"""Modular RAG MCP Server 的启动入口。"""

def main(argv: list[str] | None = None) -> int:
    """开发目录入口和安装后的 mcp-server 复用同一 stdio 实现。"""
    from src.mcp_server.server import main as server_main

    return server_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
