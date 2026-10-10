"""E2 注册表的公开行为契约。"""

import asyncio
import threading

import pytest
from jsonschema.exceptions import SchemaError
from mcp import types
from mcp.shared.exceptions import McpError

from src.mcp_server.protocol_handler import (
    ProtocolHandler,
    create_mcp_server,
    get_protocol_handler,
)


pytestmark = pytest.mark.unit


def _texts(result: types.CallToolResult) -> list[str]:
    return [item.text for item in result.content if isinstance(item, types.TextContent)]


def test_registration_defers_handler_and_keeps_schema_private() -> None:
    calls: list[dict] = []
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer", "minimum": 1}},
        "required": ["value"],
        "additionalProperties": False,
    }
    handler = ProtocolHandler()

    def capture(**arguments):
        calls.append(arguments)
        return {"received": arguments["value"]}

    handler.register_tool("capture", "Keep this schema stable", schema, capture)
    schema["properties"]["value"]["type"] = "string"
    schema["required"].clear()

    assert calls == []
    listed = handler.get_tool_schemas()
    assert calls == []
    assert len(listed) == 1
    assert listed[0].name == "capture"
    assert listed[0].description == "Keep this schema stable"
    assert listed[0].inputSchema == {
        "type": "object",
        "properties": {"value": {"type": "integer", "minimum": 1}},
        "required": ["value"],
        "additionalProperties": False,
    }
    assert "handler" not in listed[0].model_dump()

    listed[0].inputSchema["properties"]["value"]["type"] = "string"
    second_listing = handler.get_tool_schemas()
    assert second_listing[0].inputSchema["properties"]["value"]["type"] == "integer"
    assert calls == []

    result = asyncio.run(handler.execute_tool("capture", {"value": 3}))
    assert result.structuredContent == {"received": 3}
    assert calls == [{"value": 3}]


def test_registration_rejects_invalid_names_duplicates_handlers_and_schemas() -> None:
    handler = ProtocolHandler()
    valid_schema = {"type": "object"}
    handler.register_tool("existing", "", valid_schema, lambda: None)

    for invalid_name in ("", "bad name", "x" * 129):
        with pytest.raises(ValueError):
            handler.register_tool(invalid_name, "invalid", valid_schema, lambda: None)

    with pytest.raises(ValueError):
        handler.register_tool("existing", "duplicate", valid_schema, lambda: None)
    with pytest.raises(ValueError):
        handler.register_tool("not_callable", "invalid", valid_schema, object())
    with pytest.raises(ValueError):
        handler.register_tool("not_object", "invalid", {"type": "string"}, lambda: None)
    with pytest.raises(SchemaError):
        handler.register_tool(
            "malformed_schema",
            "invalid",
            {"type": "object", "properties": {"value": {"type": "not-a-json-schema-type"}}},
            lambda: None,
        )


def test_execution_supports_sync_async_and_sync_returned_awaitables() -> None:
    handler = ProtocolHandler()
    caller_thread = threading.get_ident()
    value_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def sync_tool(value: str):
        return {"value": value, "ran_in_worker": threading.get_ident() != caller_thread}

    async def async_tool(value: str):
        return {"async_value": value}

    def sync_returning_awaitable(value: str):
        async def finish():
            return {"awaited_value": value}

        return finish()

    handler.register_tool("sync", "A synchronous tool", value_schema, sync_tool)
    handler.register_tool("async", "An asynchronous tool", value_schema, async_tool)
    handler.register_tool("awaitable", "A synchronous tool returning an awaitable", value_schema, sync_returning_awaitable)

    async def invoke_all():
        return [
            await handler.execute_tool("sync", {"value": "one"}),
            await handler.execute_tool("async", {"value": "two"}),
            await handler.execute_tool("awaitable", {"value": "three"}),
        ]

    sync_result, async_result, awaitable_result = asyncio.run(invoke_all())
    assert sync_result.structuredContent == {"value": "one", "ran_in_worker": True}
    assert async_result.structuredContent == {"async_value": "two"}
    assert awaitable_result.structuredContent == {"awaited_value": "three"}


def test_execution_validates_object_schema_before_calling_handler() -> None:
    invoked: list[dict] = []
    handler = ProtocolHandler()
    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 3},
            "label": {"type": "string", "minLength": 2, "maxLength": 8},
        },
        "required": ["count", "label"],
        "additionalProperties": False,
    }
    handler.register_tool("bounded", "Bounded input", schema, lambda **args: invoked.append(args))
    invalid_arguments = (
        {"count": 2},
        {"count": "2", "label": "ok"},
        {"count": 0, "label": "ok"},
        {"count": 4, "label": "ok"},
        {"count": 2, "label": "x"},
        {"count": 2, "label": "x" * 9},
        {"count": 2, "label": "ok", "extra": True},
    )

    async def invoke_invalid_cases():
        return [await handler.execute_tool("bounded", arguments) for arguments in invalid_arguments]

    results = asyncio.run(invoke_invalid_cases())
    assert len(results) == len(invalid_arguments)
    assert all(result.isError for result in results)
    assert all("参数不符合工具 schema。" in _texts(result) for result in results)
    assert invoked == []


def test_execution_rejects_non_object_arguments_and_unknown_tools() -> None:
    handler = ProtocolHandler()
    handler.register_tool("object_only", "Object arguments", {"type": "object"}, lambda: {"ok": True})

    async def invoke_invalid_cases():
        return [
            await handler.execute_tool("object_only", ["PRIVATE_ARGUMENT"]),
            await handler.execute_tool("missing_tool", {"secret": "PRIVATE_ARGUMENT"}),
        ]

    argument_error, unknown_error = asyncio.run(invoke_invalid_cases())
    assert argument_error.isError is True
    assert unknown_error.isError is True
    assert all("PRIVATE_ARGUMENT" not in " ".join(_texts(result)) for result in (argument_error, unknown_error))


def test_execution_sanitizes_handler_errors_and_remains_usable() -> None:
    handler = ProtocolHandler()
    secret = "SENTINEL_PRIVATE_FAILURE_TEXT"
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string"}},
        "required": ["mode"],
        "additionalProperties": False,
    }

    def sometimes_fails(mode: str):
        if mode == "value":
            raise ValueError(secret)
        if mode == "type":
            raise TypeError(secret)
        if mode == "sdk":
            raise McpError(types.ErrorData(code=-32000, message=secret))
        return {"mode": mode}

    handler.register_tool("sometimes_fails", "Raises controlled probe errors", schema, sometimes_fails)

    async def invoke_failures_then_success():
        failures = [
            await handler.execute_tool("sometimes_fails", {"mode": mode})
            for mode in ("value", "type", "sdk")
        ]
        success = await handler.execute_tool("sometimes_fails", {"mode": "ok"})
        return failures, success

    failures, success = asyncio.run(invoke_failures_then_success())
    assert all(result.isError for result in failures)
    assert all(secret not in " ".join(_texts(result)) for result in failures)
    assert all(len(" ".join(_texts(result))) < 100 for result in failures)
    assert success.isError is False
    assert success.structuredContent == {"mode": "ok"}


def test_execution_returns_supported_mcp_content_shapes() -> None:
    handler = ProtocolHandler()
    object_schema = {"type": "object"}
    expected_mcp_result = types.CallToolResult(
        content=[types.TextContent(type="text", text="already MCP")],
        isError=True,
    )
    handlers = {
        "text": lambda: "plain text",
        "mapping": lambda: {"name": "锦江"},
        "content_list": lambda: [types.TextContent(type="text", text="first"), types.TextContent(type="text", text="second")],
        "mcp_result": lambda: expected_mcp_result,
        "empty": lambda: None,
        "unsupported": lambda: object(),
    }
    for name, tool_handler in handlers.items():
        handler.register_tool(name, "Probe return shape", object_schema, tool_handler)

    async def invoke_all():
        results = {}
        for name in handlers:
            results[name] = await handler.execute_tool(name, {})
        return results

    results = asyncio.run(invoke_all())
    assert results["text"].isError is False
    assert _texts(results["text"]) == ["plain text"]
    assert results["mapping"].isError is False
    assert _texts(results["mapping"]) == ['{"name": "锦江"}']
    assert results["mapping"].structuredContent == {"name": "锦江"}
    assert results["content_list"].content == handlers["content_list"]().copy()
    assert results["mcp_result"].model_dump() == expected_mcp_result.model_dump()
    assert results["empty"].isError is False
    assert results["empty"].content == []
    unsupported_text = " ".join(_texts(results["unsupported"]))
    assert results["unsupported"].isError is True
    assert len(unsupported_text) < 100
    assert "object at" not in unsupported_text


def test_server_exposes_name_version_handler_and_only_tools_capability() -> None:
    handler = ProtocolHandler(server_name="protocol-probe", server_version="2.4.1")
    server = create_mcp_server("protocol-probe", "2.4.1", handler)
    options = server.create_initialization_options()

    assert options.server_name == "protocol-probe"
    assert options.server_version == "2.4.1"
    assert get_protocol_handler(server) is handler
    assert options.capabilities.tools is not None
    assert options.capabilities.logging is None
    assert options.capabilities.prompts is None
    assert options.capabilities.resources is None
    assert options.capabilities.completions is None
    assert options.capabilities.tasks is None

    empty_server = create_mcp_server()
    assert get_protocol_handler(empty_server).get_tool_schemas() == []
