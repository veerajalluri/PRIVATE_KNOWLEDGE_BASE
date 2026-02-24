"""Tests for scripts/prepare_bi_data.py."""
import json
import sys
from pathlib import Path

import duckdb
import pytest


def _write_json(path: Path, data: list[dict]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


# Sample records matching the real data shape
HEADER_RECORDS = [
    {
        "OriginalReference": "REF001",
        "Sales Channel": "web",
        "CreatedDate": "2024-01-15",
        "Customer": "Alice",
        "NetSales": 120.0,
        "TotalSales": 144.0,
        "GrossMargin": 30.0,
    },
    {
        "OriginalReference": "REF002",
        "Sales Channel": "retail",
        "CreatedDate": "2024-02-10",
        "Customer": "Bob",
        "NetSales": 80.0,
        "TotalSales": 96.0,
        "GrossMargin": 20.0,
    },
]

LINES_RECORDS = [
    {
        "OriginalReference": "REF001",
        "styleCode": "SKU-A",
        "name": "Widget Alpha",
        "NetSales": 120.0,
        "Total Cost": 90.0,
    },
    {
        "OriginalReference": "REF002",
        "styleCode": "SKU-B",
        "name": "Widget Beta",
        "NetSales": 80.0,
        "Total Cost": 60.0,
    },
]


def _run_main(raw_folder: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_bi_data.py", str(raw_folder), str(db_path)])
    from scripts.prepare_bi_data import main
    main()


class TestPrepareBiData:
    def test_creates_orders_table(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _write_json(tmp_path / "HeaderResults.json", HEADER_RECORDS)
        db_path = tmp_path / "bi.db"

        _run_main(tmp_path, db_path, monkeypatch)

        conn = duckdb.connect(str(db_path), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()

        assert count == len(HEADER_RECORDS)

    def test_orders_table_has_correct_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _write_json(tmp_path / "HeaderResults.json", HEADER_RECORDS)
        db_path = tmp_path / "bi.db"

        _run_main(tmp_path, db_path, monkeypatch)

        conn = duckdb.connect(str(db_path), read_only=True)
        cols = {row[0] for row in conn.execute("DESCRIBE orders").fetchall()}
        conn.close()

        assert "OriginalReference" in cols
        assert "NetSales" in cols
        assert "Sales Channel" in cols

    def test_creates_order_lines_table_when_file_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _write_json(tmp_path / "HeaderResults.json", HEADER_RECORDS)
        _write_json(tmp_path / "LinesResults.json", LINES_RECORDS)
        db_path = tmp_path / "bi.db"

        _run_main(tmp_path, db_path, monkeypatch)

        conn = duckdb.connect(str(db_path), read_only=True)
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        lines_count = conn.execute("SELECT COUNT(*) FROM order_lines").fetchone()[0]
        conn.close()

        assert "orders" in tables
        assert "order_lines" in tables
        assert lines_count == len(LINES_RECORDS)

    def test_missing_lines_file_skips_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _write_json(tmp_path / "HeaderResults.json", HEADER_RECORDS)
        # No LinesResults.json
        db_path = tmp_path / "bi.db"

        _run_main(tmp_path, db_path, monkeypatch)

        conn = duckdb.connect(str(db_path), read_only=True)
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        conn.close()

        assert "orders" in tables
        assert "order_lines" not in tables

    def test_missing_header_file_exits_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # No HeaderResults.json in tmp_path
        db_path = tmp_path / "bi.db"

        with pytest.raises(SystemExit) as exc_info:
            _run_main(tmp_path, db_path, monkeypatch)

        assert exc_info.value.code == 1

    def test_creates_db_parent_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _write_json(tmp_path / "HeaderResults.json", HEADER_RECORDS)
        db_path = tmp_path / "subdir" / "nested" / "bi.db"

        _run_main(tmp_path, db_path, monkeypatch)

        assert db_path.exists()

    def test_overwrites_existing_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _write_json(tmp_path / "HeaderResults.json", HEADER_RECORDS)
        db_path = tmp_path / "bi.db"

        # Run twice — second run should replace tables, not append
        _run_main(tmp_path, db_path, monkeypatch)
        _run_main(tmp_path, db_path, monkeypatch)

        conn = duckdb.connect(str(db_path), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()

        assert count == len(HEADER_RECORDS)  # not doubled


# ── richer dataset for aggregation tests ─────────────────────────────────────
#
# orders table
#   web:       W001 £100  Jan, W002 £150  Jan, W003 £200  Feb  → total £450
#   retail:    R001 £300  Jan, R002 £250  Feb                  → total £550
#   wholesale: H001 £500  Jan                                  → total £500
#   grand total £1,500  |  Jan £1,050  |  Feb £450
#
# order_lines table
#   SKU-A: 60+90+150 = £300  (top seller)
#   SKU-B: £200
#   SKU-C: £250
#   grand total £750

RICH_HEADERS = [
    {"OriginalReference": "W001", "Sales Channel": "web",       "CreatedDate": "2024-01-10", "Customer": "Alice",   "NetSales": 100.0, "GrossMargin": 30.0},
    {"OriginalReference": "W002", "Sales Channel": "web",       "CreatedDate": "2024-01-20", "Customer": "Bob",     "NetSales": 150.0, "GrossMargin": 45.0},
    {"OriginalReference": "W003", "Sales Channel": "web",       "CreatedDate": "2024-02-05", "Customer": "Charlie", "NetSales": 200.0, "GrossMargin": 60.0},
    {"OriginalReference": "R001", "Sales Channel": "retail",    "CreatedDate": "2024-01-15", "Customer": "Dave",    "NetSales": 300.0, "GrossMargin": 90.0},
    {"OriginalReference": "R002", "Sales Channel": "retail",    "CreatedDate": "2024-02-20", "Customer": "Eve",     "NetSales": 250.0, "GrossMargin": 75.0},
    {"OriginalReference": "H001", "Sales Channel": "wholesale", "CreatedDate": "2024-01-25", "Customer": "Frank",   "NetSales": 500.0, "GrossMargin": 100.0},
]

RICH_LINES = [
    {"OriginalReference": "W001", "styleCode": "SKU-A", "name": "Widget Alpha", "NetSales":  60.0, "Total Cost":  40.0},
    {"OriginalReference": "W002", "styleCode": "SKU-A", "name": "Widget Alpha", "NetSales":  90.0, "Total Cost":  60.0},
    {"OriginalReference": "R001", "styleCode": "SKU-A", "name": "Widget Alpha", "NetSales": 150.0, "Total Cost": 100.0},
    {"OriginalReference": "W003", "styleCode": "SKU-B", "name": "Widget Beta",  "NetSales": 200.0, "Total Cost": 140.0},
    {"OriginalReference": "R002", "styleCode": "SKU-C", "name": "Widget Gamma", "NetSales": 250.0, "Total Cost": 175.0},
]


def _load_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> duckdb.DuckDBPyConnection:
    """Write rich fixtures, run prepare, return an open read-only connection."""
    _write_json(tmp_path / "HeaderResults.json", RICH_HEADERS)
    _write_json(tmp_path / "LinesResults.json", RICH_LINES)
    db_path = tmp_path / "bi.db"
    _run_main(tmp_path, db_path, monkeypatch)
    return duckdb.connect(str(db_path), read_only=True)


class TestAggregationsOnOrders:
    """Verify that SQL aggregation queries return correct results after prepare."""

    def test_total_net_sales(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        total = conn.execute('SELECT SUM("NetSales") FROM orders').fetchone()[0]
        conn.close()
        assert total == pytest.approx(1500.0)

    def test_order_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()
        assert count == 6

    def test_net_sales_by_channel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        rows = conn.execute(
            'SELECT "Sales Channel", SUM("NetSales") AS total '
            'FROM orders GROUP BY "Sales Channel" ORDER BY total DESC'
        ).fetchall()
        conn.close()
        by_channel = {ch: total for ch, total in rows}
        assert by_channel["retail"]    == pytest.approx(550.0)
        assert by_channel["wholesale"] == pytest.approx(500.0)
        assert by_channel["web"]       == pytest.approx(450.0)

    def test_order_count_by_channel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        rows = conn.execute(
            'SELECT "Sales Channel", COUNT(*) AS cnt '
            'FROM orders GROUP BY "Sales Channel" ORDER BY "Sales Channel"'
        ).fetchall()
        conn.close()
        by_channel = {ch: cnt for ch, cnt in rows}
        assert by_channel["web"]       == 3
        assert by_channel["retail"]    == 2
        assert by_channel["wholesale"] == 1

    def test_distinct_channels(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        n = conn.execute('SELECT COUNT(DISTINCT "Sales Channel") FROM orders').fetchone()[0]
        conn.close()
        assert n == 3

    def test_average_order_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        avg = conn.execute('SELECT AVG("NetSales") FROM orders').fetchone()[0]
        conn.close()
        assert avg == pytest.approx(250.0)  # 1500 / 6

    def test_max_and_min_order_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        row = conn.execute('SELECT MAX("NetSales"), MIN("NetSales") FROM orders').fetchone()
        conn.close()
        assert row[0] == pytest.approx(500.0)
        assert row[1] == pytest.approx(100.0)

    def test_january_sales(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # read_json_auto infers CreatedDate as DATE, so use date range not LIKE
        conn = _load_db(tmp_path, monkeypatch)
        total = conn.execute(
            'SELECT SUM("NetSales") FROM orders '
            "WHERE \"CreatedDate\" >= '2024-01-01' AND \"CreatedDate\" < '2024-02-01'"
        ).fetchone()[0]
        conn.close()
        assert total == pytest.approx(1050.0)  # W001+W002+R001+H001

    def test_february_sales(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        total = conn.execute(
            'SELECT SUM("NetSales") FROM orders '
            "WHERE \"CreatedDate\" >= '2024-02-01' AND \"CreatedDate\" < '2024-03-01'"
        ).fetchone()[0]
        conn.close()
        assert total == pytest.approx(450.0)  # W003+R002

    def test_top_channel_by_net_sales(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        top = conn.execute(
            'SELECT "Sales Channel" FROM orders GROUP BY "Sales Channel" '
            'ORDER BY SUM("NetSales") DESC LIMIT 1'
        ).fetchone()[0]
        conn.close()
        assert top == "retail"

    def test_gross_margin_by_channel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        rows = conn.execute(
            'SELECT "Sales Channel", SUM("GrossMargin") AS gm '
            'FROM orders GROUP BY "Sales Channel" ORDER BY "Sales Channel"'
        ).fetchall()
        conn.close()
        by_channel = {ch: gm for ch, gm in rows}
        assert by_channel["web"]       == pytest.approx(135.0)  # 30+45+60
        assert by_channel["retail"]    == pytest.approx(165.0)  # 90+75
        assert by_channel["wholesale"] == pytest.approx(100.0)

    def test_customer_lookup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        row = conn.execute(
            'SELECT "NetSales" FROM orders WHERE "Customer" = \'Frank\''
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == pytest.approx(500.0)


class TestAggregationsOnOrderLines:
    """Verify that SQL aggregation queries on order_lines return correct results."""

    def test_total_net_sales_from_lines(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        total = conn.execute('SELECT SUM("NetSales") FROM order_lines').fetchone()[0]
        conn.close()
        assert total == pytest.approx(750.0)

    def test_top_sku_by_net_sales(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        top = conn.execute(
            'SELECT "styleCode" FROM order_lines GROUP BY "styleCode" '
            'ORDER BY SUM("NetSales") DESC LIMIT 1'
        ).fetchone()[0]
        conn.close()
        assert top == "SKU-A"

    def test_net_sales_by_sku(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        rows = conn.execute(
            'SELECT "styleCode", SUM("NetSales") AS total '
            'FROM order_lines GROUP BY "styleCode" ORDER BY total DESC'
        ).fetchall()
        conn.close()
        by_sku = {sku: total for sku, total in rows}
        assert by_sku["SKU-A"] == pytest.approx(300.0)  # 60+90+150
        assert by_sku["SKU-B"] == pytest.approx(200.0)
        assert by_sku["SKU-C"] == pytest.approx(250.0)

    def test_distinct_sku_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        n = conn.execute('SELECT COUNT(DISTINCT "styleCode") FROM order_lines').fetchone()[0]
        conn.close()
        assert n == 3

    def test_gross_margin_per_sku(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        conn = _load_db(tmp_path, monkeypatch)
        rows = conn.execute(
            'SELECT "styleCode", SUM("NetSales") - SUM("Total Cost") AS gm '
            'FROM order_lines GROUP BY "styleCode" ORDER BY "styleCode"'
        ).fetchall()
        conn.close()
        by_sku = {sku: gm for sku, gm in rows}
        assert by_sku["SKU-A"] == pytest.approx(100.0)   # (60-40)+(90-60)+(150-100)
        assert by_sku["SKU-B"] == pytest.approx(60.0)    # 200-140
        assert by_sku["SKU-C"] == pytest.approx(75.0)    # 250-175

    def test_lines_join_to_orders(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify OriginalReference is preserved so lines can join to orders."""
        conn = _load_db(tmp_path, monkeypatch)
        row = conn.execute(
            'SELECT SUM(l."NetSales") '
            'FROM order_lines l '
            'JOIN orders o ON l."OriginalReference" = o."OriginalReference" '
            'WHERE o."Sales Channel" = \'web\''
        ).fetchone()
        conn.close()
        # W001 line (SKU-A £60) + W002 line (SKU-A £90) + W003 line (SKU-B £200)
        assert row[0] == pytest.approx(350.0)
