"""MCP-based BI service: spawns DuckDB MCP server, queries Claude with tools.

Replaces BISqlService. Public interface is identical:
    service.query(question: str) -> str

Architecture:
    query() [sync]
      └── asyncio.run_coroutine_threadsafe → daemon thread loop
            └── _async_query() [async]
                  ├── spawn duckdb_mcp_server subprocess (stdio)
                  ├── fetch schema once (cached for subsequent queries)
                  └── _run_agent_loop() — Claude calls execute_sql only

Dashboard mode:
    Claude returns a ```dashboard-data JSON block with KPIs + multiple charts.
    Python injects it into a fixed HTML template — identical layout every time.
"""
import asyncio
import json
import logging
import re
import sys
import time
import threading
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from injector import inject, singleton
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from private_gpt.settings.settings import Settings

_logger = logging.getLogger(__name__)

_SERVER_MODULE = "private_gpt.components.sql.duckdb_mcp_server"

_CHART_COLORS = [
    "rgba(99,102,241,0.85)",
    "rgba(16,185,129,0.85)",
    "rgba(245,158,11,0.85)",
    "rgba(239,68,68,0.85)",
    "rgba(59,130,246,0.85)",
    "rgba(168,85,247,0.85)",
    "rgba(20,184,166,0.85)",
    "rgba(251,146,60,0.85)",
]

# ── Fixed dashboard HTML template ─────────────────────────────────────────────
# Uses __DASHBOARD_JSON__ and __PALETTE_JSON__ as placeholders (not .format())
# so we don't need to escape every { } in the JavaScript.
_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:#f0f2f5;padding:24px 16px;color:#1e293b}
    .page{max-width:1100px;margin:0 auto}
    header{margin-bottom:24px}
    header h1{font-size:1.4rem;font-weight:700;color:#1e293b}
    header p{font-size:.875rem;color:#64748b;margin-top:4px}

    /* KPI row */
    .kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-bottom:24px}
    .kpi{background:#fff;border-radius:10px;padding:16px 20px;
         box-shadow:0 1px 6px rgba(0,0,0,.07)}
    .kpi-label{font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
    .kpi-value{font-size:1.6rem;font-weight:700;margin:4px 0}
    .kpi-change{font-size:.8rem;font-weight:500}
    .kpi-change.up{color:#16a34a}
    .kpi-change.down{color:#dc2626}

    /* Charts grid */
    .charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:16px}
    .chart-card{background:#fff;border-radius:10px;padding:20px 24px;
                box-shadow:0 1px 6px rgba(0,0,0,.07)}
    .chart-card h2{font-size:.95rem;font-weight:600;margin-bottom:16px;color:#334155}
    .chart-wrap{position:relative;height:280px}
  </style>
</head>
<body>
<div class="page">
  <header>
    <h1 id="dash-title"></h1>
    <p id="dash-subtitle"></p>
  </header>
  <div class="kpis" id="kpi-row"></div>
  <div class="charts" id="chart-grid"></div>
</div>
<script>
const dash = __DASHBOARD_JSON__;
const palette = __PALETTE_JSON__;

// Title
document.getElementById("dash-title").textContent = dash.title || "BI Dashboard";
document.getElementById("dash-subtitle").textContent = dash.subtitle || "";

// KPIs
const kpiRow = document.getElementById("kpi-row");
(dash.kpis || []).forEach(k => {
  const up = k.up !== false && !String(k.change || "").startsWith("-");
  kpiRow.innerHTML += `
    <div class="kpi">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      ${k.change ? `<div class="kpi-change ${up?"up":"down"}">${up?"▲":"▼"} ${k.change}</div>` : ""}
    </div>`;
});

// Charts — pass 1: build all HTML at once (avoids innerHTML+= destroying canvas elements)
const grid = document.getElementById("chart-grid");
const charts = dash.charts || [];
grid.innerHTML = charts.map((cfg, idx) =>
  '<div class="chart-card">' +
  '<h2>' + (cfg.title || "") + '</h2>' +
  '<div class="chart-wrap"><canvas id="chart-' + idx + '"></canvas></div>' +
  '</div>'
).join("");

// Charts — pass 2: attach Chart.js to each canvas (DOM is fully built now)
charts.forEach((cfg, idx) => {
  const isPie = cfg.type === "pie" || cfg.type === "doughnut";
  cfg.data.datasets.forEach((ds, di) => {
    if (isPie) {
      ds.backgroundColor = palette;
      ds.borderColor = "#fff";
      ds.borderWidth = 2;
    } else {
      ds.backgroundColor = ds.backgroundColor || palette[di % palette.length];
      ds.borderColor = palette[di % palette.length].replace("0.85","1");
      ds.borderWidth = 1;
    }
  });
  new Chart(document.getElementById("chart-" + idx), {
    type: cfg.type || "bar",
    data: cfg.data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: isPie ? "right" : "top" },
        tooltip: { mode: isPie ? "point" : "index", intersect: false }
      },
      scales: isPie ? {} : {
        x: { title: { display: !!cfg.x_label, text: cfg.x_label||"" }, grid: { display:false } },
        y: { title: { display: !!cfg.y_label, text: cfg.y_label||"" }, beginAtZero:true,
             grid: { color:"rgba(0,0,0,.05)" } }
      }
    }
  });
});
</script>
</body>
</html>"""

# ── System prompts ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_TEMPLATE = """\
You are a business intelligence analyst with direct access to a DuckDB database.

Database schema:
{schema}

When answering questions about the data:
1. Write a precise DuckDB SELECT query and call execute_sql to run it.
2. Interpret the results and provide a clear, concise answer.
3. Include the SQL you used in a fenced sql code block.

Do not make up data. If the query returns no rows, say so clearly.
"""

_SYSTEM_PROMPT_DASHBOARD_TEMPLATE = """\
You are a business intelligence analyst with direct access to a DuckDB database.

Database schema:
{schema}

When answering questions about the data:
1. Run as many execute_sql calls as needed to gather all data for the dashboard.
2. Return the results as a single ```dashboard-data JSON block (no HTML, no extra text before it).

The JSON must follow this exact schema:
```dashboard-data
{{
  "title": "Dashboard title",
  "subtitle": "One-sentence description or date range",
  "kpis": [
    {{"label": "Metric name", "value": "£12,345", "change": "+5.2%", "up": true}}
  ],
  "charts": [
    {{
      "title": "Chart title",
      "type": "bar",
      "x_label": "X axis",
      "y_label": "Y axis",
      "data": {{
        "labels": ["A", "B"],
        "datasets": [{{"label": "Series", "data": [100, 200]}}]
      }}
    }},
    {{
      "title": "Mix chart",
      "type": "pie",
      "data": {{
        "labels": ["A", "B"],
        "datasets": [{{"data": [60, 40]}}]
      }}
    }},
    {{
      "title": "Trend chart",
      "type": "line",
      "x_label": "Month",
      "y_label": "Value",
      "data": {{
        "labels": ["Jan", "Feb"],
        "datasets": [{{"label": "Series", "data": [100, 120]}}]
      }}
    }}
  ]
}}
```

Rules:
- Include 3–5 KPI cards and 3–5 charts (bar, pie/doughnut, line as appropriate).
- After the ```dashboard-data block write one short plain-text summary sentence.
- Do NOT generate any HTML. Do not make up data.
"""

_CHART_KEYWORDS = {"chart", "graph", "plot", "visual", "html", "report", "dashboard", "insight", "overview"}


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

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="mcp-bi-loop"
        )
        self._loop_thread.start()

        self._schema_cache: str | None = None

        _logger.info("MCPBiService: initialized with model=%s", self._model)

    # ── public API (synchronous) ───────────────────────────────────────────────

    def query(self, question: str) -> str:
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
            env=None,
        )
        t0 = time.monotonic()
        async with stdio_client(server_params) as (read, write):
            _logger.info("MCPBiService: subprocess started in %.1fs", time.monotonic() - t0)
            async with ClientSession(read, write) as session:
                await session.initialize()
                _logger.info("MCPBiService: session initialized in %.1fs", time.monotonic() - t0)

                if self._schema_cache is None:
                    schema_result = await session.call_tool("get_schema", {})
                    self._schema_cache = "\n".join(
                        c.text for c in schema_result.content if hasattr(c, "text")
                    )
                    _logger.info(
                        "MCPBiService: schema fetched in %.1fs (%d chars)",
                        time.monotonic() - t0, len(self._schema_cache),
                    )

                wants_dashboard = any(kw in question.lower() for kw in _CHART_KEYWORDS)
                if wants_dashboard:
                    system_prompt = _SYSTEM_PROMPT_DASHBOARD_TEMPLATE.format(
                        schema=self._schema_cache
                    )
                    max_tokens = 4096
                else:
                    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                        schema=self._schema_cache
                    )
                    max_tokens = 2048

                all_tools = await self._get_mcp_tools(session)
                sql_tools = [t for t in all_tools if t["name"] != "get_schema"]

                result = await self._run_agent_loop(
                    session, question, sql_tools, system_prompt, max_tokens
                )
                _logger.info("MCPBiService: total query time %.1fs", time.monotonic() - t0)
                return result

    async def _get_mcp_tools(self, session: ClientSession) -> list[dict[str, Any]]:
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
        system_prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        client = AsyncAnthropic(api_key=self._api_key)
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        while True:
            t_call = time.monotonic()
            response = await client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tools,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )
            _logger.info(
                "MCPBiService: Claude stop_reason=%s blocks=%d (%.1fs)",
                response.stop_reason, len(response.content), time.monotonic() - t_call,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason in ("end_turn", "max_tokens"):
                for block in response.content:
                    if hasattr(block, "text"):
                        return self._render_dashboard_or_return(block.text)
                return "(No text response from Claude)"

            if response.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    _logger.info("MCPBiService: tool_use name=%s", block.name)
                    mcp_result = await session.call_tool(block.name, block.input)
                    result_text = "\n".join(
                        c.text for c in mcp_result.content if hasattr(c, "text")
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            _logger.warning("MCPBiService: unexpected stop_reason=%s", response.stop_reason)
            break

        return "Claude stopped unexpectedly. Please try again."

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_dashboard_or_return(self, text: str) -> str:
        """Parse dashboard-data JSON block and render into fixed HTML template.
        Falls back to saving raw HTML if present. Returns only a clickable link."""

        # ── Path 1: structured dashboard-data block ───────────────────────────
        match = re.search(r"```dashboard-data\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            try:
                dash = json.loads(match.group(1))
                html = (
                    _DASHBOARD_TEMPLATE
                    .replace("__TITLE__", dash.get("title", "BI Dashboard"))
                    .replace("__DASHBOARD_JSON__", json.dumps(dash))
                    .replace("__PALETTE_JSON__", json.dumps(_CHART_COLORS))
                )
                link = self._save_report(html)
                summary = text[match.end():].strip()
                return f"{summary}\n\n---\n**View dashboard:** [Open report]({link})"
            except json.JSONDecodeError as exc:
                _logger.warning("MCPBiService: dashboard JSON parse error: %s", exc)

        # ── Path 2: raw HTML fallback ─────────────────────────────────────────
        lower = text.lower()
        html_start = lower.find("<!doctype html>")
        if html_start == -1:
            html_start = lower.find("<html")
        if html_start != -1:
            prefix = text[:html_start].strip()
            html = text[html_start:]
            fence = html.find("\n```")
            if fence != -1:
                html = html[:fence]
            link = self._save_report(html)
            return f"{prefix}\n\n---\n**View report:** [Open report]({link})"

        return text

    def _save_report(self, html: str) -> str:
        reports_dir = Path(__file__).parents[3] / "local_data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"report_{int(time.time())}.html"
        (reports_dir / filename).write_text(html, encoding="utf-8")
        _logger.info("MCPBiService: saved report %s", filename)
        return f"http://localhost:8001/reports/{filename}"
