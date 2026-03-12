"""MCP-based BI service: spawns DuckDB MCP server, queries Claude with tools.

Replaces BISqlService. Public interface is identical:
    service.query(question: str) -> str

Architecture:
    query() [sync]
      └── asyncio.run_coroutine_threadsafe → daemon thread loop
            └── _async_query() [async]
                  ├── spawn duckdb_mcp_server subprocess (stdio)
                  ├── open ClientSession
                  └── _run_agent_loop() — Claude calls get_schema + execute_sql
"""
import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import anthropic
from injector import inject, singleton
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from private_gpt.settings.settings import Settings

_logger = logging.getLogger(__name__)

_SERVER_MODULE = "private_gpt.components.sql.duckdb_mcp_server"

_SYSTEM_PROMPT = """\
You are a business intelligence analyst with direct access to a DuckDB database.

When answering questions about the data:
1. Always call get_schema first to understand the available tables and columns.
2. Write a precise DuckDB SELECT query and call execute_sql to run it.
3. Interpret the results and provide a clear, concise answer.
4. Include the SQL you used in a fenced sql code block.

Do not make up data. If the query returns no rows, say so clearly.
"""


@singleton
class MCPBiService:
    """Text-to-SQL via Claude + DuckDB MCP server."""

    @inject
    def __init__(self, settings: Settings) -> None:
        if settings.anthropic is None:
            raise RuntimeError(
                "MCPBiService requires [anthropic] settings. "
                "Add 'anthropic: {api_key: ..., model: ...}' to settings-bi.yaml "
                "and set ANTHROPIC_API_KEY."
            )
        self._api_key = settings.anthropic.api_key
        self._model = settings.anthropic.model

        # Dedicated daemon thread with its own event loop for sync→async bridging.
        # This avoids conflicts with uvicorn's running event loop (asyncio.run()
        # would fail if called from within an already-running loop).
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="mcp-bi-loop"
        )
        self._loop_thread.start()
        _logger.info(
            "MCPBiService: initialized with model=%s", self._model
        )

    # ── public API (synchronous) ───────────────────────────────────────────────

    def query(self, question: str) -> str:
        """Translate a natural-language question to SQL via Claude + MCP tools."""
        if not question or not question.strip():
            return "Please ask a question about the business data."
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_query(question.strip()), self._loop
            )
            return future.result(timeout=120)
        except TimeoutError:
            _logger.error("MCPBiService: query timed out after 120s")
            return "Query timed out. The database or model took too long to respond."
        except Exception as exc:
            _logger.exception("MCPBiService: unexpected error during query")
            return f"An error occurred: {exc}"

    # ── async internals ────────────────────────────────────────────────────────

    async def _async_query(self, question: str) -> str:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", _SERVER_MODULE],
            env=None,  # inherit parent environment
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await self._get_mcp_tools(session)
                return await self._run_agent_loop(session, question, tools)

    async def _get_mcp_tools(self, session: ClientSession) -> list[dict[str, Any]]:
        """Fetch tool definitions from MCP server and convert to Anthropic format."""
        result = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]

    async def _run_agent_loop(
        self,
        session: ClientSession,
        question: str,
        tools: list[dict[str, Any]],
    ) -> str:
        """Run the Claude agentic loop until the model produces a final answer."""
        client = anthropic.Anthropic(api_key=self._api_key)
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        while True:
            response = client.messages.create(
                model=self._model,
                max_tokens=8096,
                system=_SYSTEM_PROMPT,
                tools=tools,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )
            _logger.info(
                "MCPBiService: Claude stop_reason=%s blocks=%d",
                response.stop_reason,
                len(response.content),
            )

            # Append assistant turn (preserves full content list for context)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason in ("end_turn", "max_tokens"):
                for block in response.content:
                    if hasattr(block, "text"):
                        return self._maybe_save_html(block.text)
                return "(No text response from Claude)"

            if response.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    _logger.info(
                        "MCPBiService: tool_use name=%s input=%s",
                        block.name,
                        block.input,
                    )
                    mcp_result = await session.call_tool(block.name, block.input)
                    result_text = "\n".join(
                        c.text for c in mcp_result.content if hasattr(c, "text")
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                continue

            _logger.warning(
                "MCPBiService: unexpected stop_reason=%s", response.stop_reason
            )
            break

        return "Claude stopped unexpectedly. Please try again."

    def _maybe_save_html(self, text: str) -> str:
        """If the response contains an HTML page, save it and append a link."""
        # Extract the HTML block from the response (handles fenced or raw HTML)
        html = None
        if "<!DOCTYPE html>" in text or "<html" in text.lower():
            # Try to extract from markdown code fence first
            import re
            match = re.search(r"```(?:html)?\s*(<!DOCTYPE html>.*?)</?\s*```", text, re.DOTALL | re.IGNORECASE)
            if match:
                html = match.group(1).strip()
            else:
                # Raw HTML in response
                start = text.lower().find("<!doctype html>")
                if start == -1:
                    start = text.lower().find("<html")
                html = text[start:].strip()

        if not html:
            return text

        reports_dir = Path(__file__).parents[3] / "local_data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"report_{int(time.time())}.html"
        report_path = reports_dir / filename
        report_path.write_text(html, encoding="utf-8")
        _logger.info("MCPBiService: saved HTML report to %s", report_path)

        link = f"http://localhost:8001/reports/{filename}"
        return f"{text}\n\n---\n**View report:** [{filename}]({link})"
