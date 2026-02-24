"""Tests for BISqlService, _extract_sql, and _format_result."""
import json
from unittest.mock import MagicMock

import duckdb
import pytest

from private_gpt.components.sql.bi_sql_service import (
    BISqlService,
    _extract_sql,
    _format_result,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _orders_conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with a small orders table for testing."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE orders (channel VARCHAR, net_sales DOUBLE)")
    conn.execute(
        "INSERT INTO orders VALUES ('web', 100.0), ('retail', 200.0), ('wholesale', 50.0)"
    )
    return conn


def _make_service(
    llm_responses: list[str],
    conn: duckdb.DuckDBPyConnection | None = None,
) -> BISqlService:
    """Build a BISqlService with a mocked LLM and the given connection.

    Bypasses DI by setting internal attributes directly.
    """
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [MagicMock(text=r) for r in llm_responses]

    svc = object.__new__(BISqlService)
    svc._llm = mock_llm
    svc._conn = conn or _orders_conn()
    svc._schema_ddl = "CREATE TABLE orders (channel VARCHAR, net_sales DOUBLE);"
    return svc


# ── _extract_sql ─────────────────────────────────────────────────────────────

class TestExtractSql:
    def test_plain_sql_returned_as_is(self):
        sql = "SELECT * FROM orders"
        assert _extract_sql(sql) == sql

    def test_strips_sql_fenced_block(self):
        raw = "```sql\nSELECT * FROM orders;\n```"
        assert _extract_sql(raw) == "SELECT * FROM orders;"

    def test_strips_plain_fenced_block(self):
        raw = "```\nSELECT 1;\n```"
        assert _extract_sql(raw) == "SELECT 1;"

    def test_stops_at_first_semicolon(self):
        raw = "SELECT * FROM orders; SELECT 1;"
        assert _extract_sql(raw) == "SELECT * FROM orders;"

    def test_no_semicolon_returns_full_text(self):
        raw = "SELECT channel FROM orders"
        assert _extract_sql(raw) == raw

    def test_strips_surrounding_whitespace(self):
        raw = "  \n  SELECT 1;  \n  "
        assert _extract_sql(raw) == "SELECT 1;"

    def test_fenced_block_with_extra_text_before(self):
        raw = "Here is the SQL:\n```sql\nSELECT COUNT(*) FROM orders;\n```"
        assert _extract_sql(raw) == "SELECT COUNT(*) FROM orders;"


# ── _format_result ────────────────────────────────────────────────────────────

class TestFormatResult:
    def test_returns_markdown_table_with_header(self):
        conn = _orders_conn()
        result = _format_result(conn, "SELECT channel, net_sales FROM orders ORDER BY net_sales")
        lines = result.strip().split("\n")
        assert "channel" in lines[0]
        assert "net_sales" in lines[0]
        assert "wholesale" in result
        assert "web" in result
        assert "retail" in result

    def test_empty_result_returns_no_rows_message(self):
        conn = _orders_conn()
        result = _format_result(conn, "SELECT * FROM orders WHERE net_sales > 9999")
        assert result == "*(no rows returned)*"

    def test_raises_on_invalid_sql(self):
        conn = _orders_conn()
        with pytest.raises(Exception):
            _format_result(conn, "SELECT * FROM nonexistent_table_xyz")

    def test_raises_on_syntax_error(self):
        conn = _orders_conn()
        with pytest.raises(Exception):
            _format_result(conn, "THIS IS NOT SQL")

    def test_column_widths_fit_values(self):
        conn = _orders_conn()
        result = _format_result(conn, "SELECT channel FROM orders ORDER BY channel")
        # Each line should have consistent column alignment
        lines = [l for l in result.split("\n") if l.strip()]
        # Header and separator should have same width structure
        assert len(lines) >= 2

    def test_aggregation_query(self):
        conn = _orders_conn()
        result = _format_result(conn, "SELECT SUM(net_sales) AS total FROM orders")
        assert "350" in result  # 100 + 200 + 50


# ── BISqlService.query ────────────────────────────────────────────────────────

class TestBISqlServiceQuery:
    def test_empty_question_returns_placeholder(self):
        svc = _make_service(llm_responses=[])
        result = svc.query("")
        assert "Please ask a question" in result

    def test_whitespace_question_returns_placeholder(self):
        svc = _make_service(llm_responses=[])
        result = svc.query("   ")
        assert "Please ask a question" in result

    def test_successful_query_returns_sql_and_results(self):
        svc = _make_service(
            llm_responses=["SELECT channel, net_sales FROM orders ORDER BY net_sales;"]
        )
        result = svc.query("Show me all channels by net sales")
        assert "**SQL:**" in result
        assert "**Results:**" in result
        assert "wholesale" in result

    def test_fenced_sql_from_llm_is_handled(self):
        svc = _make_service(
            llm_responses=["```sql\nSELECT SUM(net_sales) AS total FROM orders;\n```"]
        )
        result = svc.query("What is total net sales?")
        assert "**SQL:**" in result
        assert "350" in result

    def test_retries_once_on_sql_error(self):
        svc = _make_service(
            llm_responses=[
                "SELECT * FROM nonexistent_table;",               # attempt 1 — bad
                "SELECT SUM(net_sales) AS total FROM orders;",    # attempt 2 — good
            ]
        )
        result = svc.query("What is total net sales?")
        assert "**SQL:**" in result
        assert "350" in result
        # LLM was called twice
        assert svc._llm.complete.call_count == 2

    def test_retry_prompt_includes_error_message(self):
        svc = _make_service(
            llm_responses=[
                "SELECT * FROM nonexistent_table;",
                "SELECT COUNT(*) FROM orders;",
            ]
        )
        svc.query("How many orders?")
        # Second call's prompt should contain the error from the first attempt
        second_call_prompt = svc._llm.complete.call_args_list[1][0][0]
        assert "previous SQL attempt failed" in second_call_prompt

    def test_exhausted_retries_returns_error_message(self):
        svc = _make_service(
            llm_responses=[
                "SELECT * FROM bad_table_1;",
                "SELECT * FROM bad_table_2;",
            ]
        )
        result = svc.query("What is revenue?")
        assert "Could not execute SQL after 2 attempts" in result
        assert "Error:" in result

    def test_llm_exception_returns_error_message(self):
        svc = _make_service(llm_responses=[])
        svc._llm.complete.side_effect = RuntimeError("Ollama timeout")
        result = svc.query("What is total revenue?")
        assert "Could not generate SQL" in result
        assert "Ollama timeout" in result

    def test_no_rows_returns_no_rows_message(self):
        svc = _make_service(
            llm_responses=["SELECT * FROM orders WHERE net_sales > 99999;"]
        )
        result = svc.query("Find orders over 99999")
        assert "no rows returned" in result

    def test_result_contains_generated_sql(self):
        sql = "SELECT channel FROM orders ORDER BY channel;"
        svc = _make_service(llm_responses=[sql])
        result = svc.query("List channels")
        assert sql in result
