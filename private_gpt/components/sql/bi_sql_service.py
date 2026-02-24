"""Text-to-SQL service for conversational BI using DuckDB.

Workflow:
1. Open the persistent DuckDB file written by prepare_bi_data.py.
2. Generate CREATE TABLE DDL strings for each table (used as schema context).
3. On each query: build an NSQL prompt → call the LLM for SQL → execute against
   DuckDB → format the result as readable text.
4. On SQL execution error, retry once with the error message appended to the
   prompt so the model can self-correct.

No vector embeddings or vector store are involved.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from injector import inject, singleton

from private_gpt.components.llm.llm_component import LLMComponent

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

_DB_PATH = Path("local_data/bi.db")

_NSQL_PROMPT = """\
### Instruction:
Your task is to generate valid duckdb SQL to answer the following question.

### Input:
Here is the database schema that the SQL query will run on:
{schema_ddl}

### Question:
{question}

### Response (use duckdb syntax, no explanation):
"""


def _extract_sql(raw: str) -> str:
    """Extract the first SQL statement from the model output.

    Handles both plain SQL and markdown-fenced blocks (```sql ... ``` or ``` ... ```).
    """
    text = raw.strip()

    # Strip markdown fences if present
    if "```" in text:
        parts = text.split("```")
        # parts[1] is the content inside the first fence pair
        if len(parts) > 1:
            text = parts[1].lstrip("sql").strip()

    # Stop at first semicolon (inclusive)
    if ";" in text:
        text = text[: text.index(";") + 1]

    return text.strip()


def _format_result(conn: "duckdb.DuckDBPyConnection", sql: str) -> str:
    """Execute SQL and return a plain-text formatted result.

    Raises duckdb.Error (or any exception) on bad SQL so the caller can retry.
    """
    rel = conn.execute(sql)
    cols = [desc[0] for desc in rel.description]
    rows = rel.fetchall()

    if not rows:
        return "*(no rows returned)*"

    # Build a simple markdown table
    col_widths = [
        max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
        for i, c in enumerate(cols)
    ]
    sep = " | ".join("-" * w for w in col_widths)
    header = " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols))
    body = "\n".join(
        " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))
        for row in rows
    )
    return f"{header}\n{sep}\n{body}"


@singleton
class BISqlService:
    """Text-to-SQL service backed by a persistent DuckDB file."""

    @inject
    def __init__(self, llm_component: LLMComponent) -> None:
        self._llm = llm_component.llm
        self._conn: "duckdb.DuckDBPyConnection | None" = None
        self._schema_ddl: str = ""

    # ── internal setup ───────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._conn is not None:
            return

        if not _DB_PATH.exists():
            logger.warning(
                "BISqlService: %s not found. Run 'make prepare' first.",
                _DB_PATH,
            )
            self._conn = duckdb.connect(":memory:")
            self._schema_ddl = "-- no tables available"
            return

        self._conn = duckdb.connect(str(_DB_PATH.resolve()), read_only=True)

        tables = [row[0] for row in self._conn.execute("SHOW TABLES").fetchall()]
        if not tables:
            logger.warning("BISqlService: DuckDB file exists but has no tables.")
            self._schema_ddl = "-- no tables available"
            return

        ddl_parts: list[str] = []
        for table_name in tables:
            cols = self._conn.execute(f"DESCRIBE {table_name}").fetchall()
            col_defs = ", ".join(f'"{c[0]}" {c[1]}' for c in cols)
            ddl_parts.append(f"CREATE TABLE {table_name} ({col_defs});")
            logger.info("BISqlService: registered table '%s' (%d columns)", table_name, len(cols))

        self._schema_ddl = "\n".join(ddl_parts)
        logger.info("BISqlService: ready with %d table(s) from %s.", len(tables), _DB_PATH)

    # ── public API ───────────────────────────────────────────────────────────

    def query(self, question: str) -> str:
        """Translate *question* to SQL, execute it, and return formatted results.

        Retries once if DuckDB returns an error, feeding the error message back
        to the model so it can self-correct.
        """
        self._ensure_connected()
        assert self._conn is not None

        if not question or not question.strip():
            return "Please ask a question about the business data."

        prompt = _NSQL_PROMPT.format(
            schema_ddl=self._schema_ddl,
            question=question.strip(),
        )

        last_error: str | None = None
        sql = ""

        for attempt in range(2):
            if last_error:
                # Append error feedback so the model can self-correct on retry
                prompt += (
                    f"\n\n-- The previous SQL attempt failed with: {last_error}\n"
                    "-- Corrected SQL:\n"
                )

            try:
                raw_sql = self._llm.complete(prompt).text
            except Exception as exc:
                logger.exception("BISqlService: LLM completion failed")
                return f"Could not generate SQL: {exc}"

            sql = _extract_sql(raw_sql)
            logger.info("BISqlService (attempt %d): %s", attempt + 1, sql)

            try:
                result_table = _format_result(self._conn, sql)
                return (
                    f"**SQL:**\n```sql\n{sql}\n```\n\n"
                    f"**Results:**\n```\n{result_table}\n```"
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "BISqlService: SQL error on attempt %d: %s", attempt + 1, exc
                )

        return (
            f"Could not execute SQL after 2 attempts.\n\n"
            f"Last SQL:\n```sql\n{sql}\n```\n\n"
            f"Error: {last_error}"
        )
