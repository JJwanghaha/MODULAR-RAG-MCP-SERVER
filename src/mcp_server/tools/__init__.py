"""MCP Tools 包。"""
def register_default_tools(protocol_handler, settings=None, *, data_dir=None, embedding_client=None) -> None:
    """登记三个真实工具；仅保存配置/函数引用，不打开存储或创建模型。"""
    from src.mcp_server.tools.query_knowledge_hub import register_tool as register_query
    from src.mcp_server.tools.list_collections import register_tool as register_collections
    from src.mcp_server.tools.get_document_summary import register_tool as register_summary

    register_query(protocol_handler, settings, data_dir=data_dir, embedding_client=embedding_client)
    register_collections(protocol_handler, settings, data_dir=data_dir)
    register_summary(protocol_handler, settings, data_dir=data_dir)
