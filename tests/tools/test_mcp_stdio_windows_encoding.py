"""MCP stdio transport uses the SDK's strict encoding error handler on
POSIX and ``replace`` on Windows, where pipe reads are not guaranteed to
be UTF-8-aligned and the strict default crashes the connection on a bad
byte at a chunk boundary (#46099).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_server():
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("test_srv")
    server._sampling = None
    return server


def _stdio_client_factory():
    """stdio_client stand-in: the call returns a context manager that
    raises on enter, so the test captures the StdioServerParameters kwargs
    before the real transport is touched.
    """
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(
        side_effect=RuntimeError("sentinel: stdio_client not entered in test")
    )
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(side_effect=lambda *a, **kw: cm)


def _patch_transport(monkeypatch, mcp_tool):
    monkeypatch.setattr(
        "tools.osv_check.check_package_for_malware", lambda cmd, args: None
    )
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "stdio_client", _stdio_client_factory())
    monkeypatch.setattr(mcp_tool, "_MCP_NOTIFICATION_TYPES", False)


def _run_and_capture(monkeypatch, platform):
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool.sys, "platform", platform)

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(mcp_tool, "StdioServerParameters", _capture)
    _patch_transport(monkeypatch, mcp_tool)

    server = _make_server()
    with patch.object(mcp_tool, "_resolve_stdio_command", lambda c, e: (c, e)), \
         patch.object(mcp_tool, "_build_safe_env", lambda e: e or {}), \
         patch.object(mcp_tool, "_snapshot_child_pids", lambda: set()), \
         patch.object(mcp_tool, "_get_mcp_stderr_log", lambda: MagicMock()), \
         patch.object(mcp_tool, "_write_stderr_log_header", lambda n: None), \
         patch("builtins.open", MagicMock()):
        try:
            asyncio.run(server._run_stdio({"command": "echo", "args": []}))
        except RuntimeError:
            pass

    return captured


class TestEncodingErrorHandler:
    def test_windows_uses_replace(self, monkeypatch):
        captured = _run_and_capture(monkeypatch, "win32")
        assert captured.get("encoding_error_handler") == "replace"

    def test_posix_uses_strict(self, monkeypatch):
        captured = _run_and_capture(monkeypatch, "linux")
        assert captured.get("encoding_error_handler") == "strict"

    def test_replace_is_accepted_by_sdk_literal(self):
        """``replace`` is in the SDK's Literal; ``surrogateescape`` is not."""
        from mcp.client.stdio import StdioServerParameters

        StdioServerParameters(command="echo", encoding_error_handler="replace")
