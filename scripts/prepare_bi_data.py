#!/usr/bin/env python3
"""Load raw BI JSON files directly into DuckDB.

Uses DuckDB's read_json_auto() to ingest all records from:
  HeaderResults.json  →  table 'orders'
  LinesResults.json   →  table 'order_lines'

No sampling, no pre-aggregation. The text-to-SQL service runs
arbitrary SQL against the full dataset at query time. DuckDB
handles 100K–500K rows in seconds without Python buffering.

Usage:
    python scripts/prepare_bi_data.py [raw_folder] [db_path]

    raw_folder   default: local_data/uploads
    db_path      default: local_data/bi.db
"""
import logging
import sys
from pathlib import Path

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    raw_folder = Path(sys.argv[1] if len(sys.argv) > 1 else "local_data/uploads")
    db_path = Path(sys.argv[2] if len(sys.argv) > 2 else "local_data/bi.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    header_file = raw_folder / "HeaderResults.json"
    lines_file = raw_folder / "LinesResults.json"

    if not header_file.exists():
        logger.error("HeaderResults.json not found in %s — cannot continue.", raw_folder)
        sys.exit(1)

    conn = duckdb.connect(str(db_path))

    # Load all order headers
    logger.info("Loading %s (%.0f MB) …", header_file.name, header_file.stat().st_size / 1e6)
    conn.execute(
        f"CREATE OR REPLACE TABLE orders AS SELECT * FROM read_json_auto('{header_file.resolve()}')"
    )
    n = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    logger.info("  → %d rows in 'orders'", n)

    # Load all order lines (optional)
    if lines_file.exists():
        logger.info("Loading %s (%.0f MB) …", lines_file.name, lines_file.stat().st_size / 1e6)
        conn.execute(
            f"CREATE OR REPLACE TABLE order_lines AS SELECT * FROM read_json_auto('{lines_file.resolve()}')"
        )
        n = conn.execute("SELECT COUNT(*) FROM order_lines").fetchone()[0]
        logger.info("  → %d rows in 'order_lines'", n)
    else:
        logger.warning("LinesResults.json not found in %s — skipping order lines.", raw_folder)

    tables = conn.execute("SHOW TABLES").fetchall()
    total_rows = sum(
        conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0] for t in tables
    )
    conn.close()
    logger.info("Done. %d table(s), %d total rows → %s", len(tables), total_rows, db_path)


if __name__ == "__main__":
    main()
