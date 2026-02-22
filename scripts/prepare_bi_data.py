#!/usr/bin/env python3
"""Pre-aggregate raw BI JSON files into compact summary documents for RAG.

Why: 678K raw records cannot be embedded individually on CPU (it would take
hours and RAG top-k=10 would only see 0.001% of data for aggregation queries).

This script reads the raw HeaderResults.json and LinesResults.json, computes
meaningful aggregations, and writes ~200 small JSON files to data/aggregated/.
The ingest_bi_json.py script then embeds those summaries — not the raw data.

Usage:
    python scripts/prepare_bi_data.py [raw_folder] [out_folder]

    raw_folder  default: data/uploads
    out_folder  default: data/aggregated
"""
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _f(val, default: float = 0.0) -> float:
    """Safe float cast."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _month(date_str: str) -> str:
    """Extract YYYY-MM from an ISO date string; return 'unknown' on failure."""
    try:
        return datetime.fromisoformat(date_str).strftime("%Y-%m")
    except Exception:
        return "unknown"


def _pct(numerator: float, denominator: float) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def _save(data: list | dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("  Wrote %s (%d records)", path.name, len(data) if isinstance(data, list) else 1)


# ── HeaderResults aggregations ────────────────────────────────────────────────

def aggregate_headers(records: list[dict], out: Path) -> None:
    logger.info("Aggregating %d order headers …", len(records))

    # 1. Overall summary
    overall: dict = defaultdict(float)
    overall["record_type"] = "overall_order_summary"
    overall["total_orders"] = len(records)
    channels: set[str] = set()
    customers: set[str] = set()
    dates: list[str] = []

    for r in records:
        overall["total_net_sales"] += _f(r.get("NetSales"))
        overall["total_vat"] += _f(r.get("VAT"))
        overall["total_sales_inc_vat"] += _f(r.get("TotalSales"))
        overall["total_qty"] += _f(r.get("Total Qty"))
        overall["total_discount"] += _f(r.get("DiscountTotal"))
        overall["total_material_cost"] += _f(r.get("Material Cost"))
        overall["total_packaging_cost"] += _f(r.get("Packaging Cost"))
        overall["total_labour_cost"] += _f(r.get("Labour Cost"))
        overall["total_distribution_cost"] += _f(r.get("Distribution Cost"))
        overall["total_wastage_cost"] += _f(r.get("Wastage Cost"))
        overall["total_gross_margin"] += _f(r.get("GrossMargin"))
        overall["total_shipping_net"] += _f(r.get("ShippingNet"))
        overall["total_commission"] += _f(r.get("Amazon/ebay Commission"))
        channels.add(str(r.get("Sales Channel") or "Unknown"))
        if r.get("Customer"):
            customers.add(str(r["Customer"]))
        if r.get("CreatedDate"):
            dates.append(r["CreatedDate"])

    overall["gross_margin_pct"] = _pct(overall["total_gross_margin"], overall["total_net_sales"])
    overall["average_order_value"] = round(overall["total_net_sales"] / len(records), 2) if records else 0
    overall["unique_channels"] = len(channels)
    overall["unique_customers"] = len(customers)
    if dates:
        overall["date_from"] = min(dates)
        overall["date_to"] = max(dates)
    _save([dict(overall)], out / "header_overall_summary.json")

    # 2. By Sales Channel
    by_channel: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in records:
        ch = str(r.get("Sales Channel") or "Unknown")
        g = by_channel[ch]
        g["sales_channel"] = ch
        g["record_type"] = "channel_summary"
        g["order_count"] += 1
        g["net_sales"] += _f(r.get("NetSales"))
        g["total_sales_inc_vat"] += _f(r.get("TotalSales"))
        g["total_qty"] += _f(r.get("Total Qty"))
        g["gross_margin"] += _f(r.get("GrossMargin"))
        g["total_discount"] += _f(r.get("DiscountTotal"))
        g["material_cost"] += _f(r.get("Material Cost"))
        g["distribution_cost"] += _f(r.get("Distribution Cost"))
        g["commission"] += _f(r.get("Amazon/ebay Commission"))

    channel_rows = []
    for ch, g in by_channel.items():
        g["gross_margin_pct"] = _pct(g["gross_margin"], g["net_sales"])
        g["average_order_value"] = round(g["net_sales"] / g["order_count"], 2) if g["order_count"] else 0
        channel_rows.append(dict(g))
    channel_rows.sort(key=lambda x: x["net_sales"], reverse=True)
    _save(channel_rows, out / "header_by_channel.json")

    # 3. By Month
    by_month: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in records:
        mo = _month(str(r.get("CreatedDate") or ""))
        g = by_month[mo]
        g["month"] = mo
        g["record_type"] = "monthly_summary"
        g["order_count"] += 1
        g["net_sales"] += _f(r.get("NetSales"))
        g["total_sales_inc_vat"] += _f(r.get("TotalSales"))
        g["total_qty"] += _f(r.get("Total Qty"))
        g["gross_margin"] += _f(r.get("GrossMargin"))
        g["total_discount"] += _f(r.get("DiscountTotal"))
        g["distribution_cost"] += _f(r.get("Distribution Cost"))

    month_rows = []
    for mo, g in by_month.items():
        g["gross_margin_pct"] = _pct(g["gross_margin"], g["net_sales"])
        g["average_order_value"] = round(g["net_sales"] / g["order_count"], 2) if g["order_count"] else 0
        month_rows.append(dict(g))
    month_rows.sort(key=lambda x: x["month"])
    _save(month_rows, out / "header_by_month.json")

    # 4. By Customer (top 100 by net sales)
    by_cust: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in records:
        cust = str(r.get("Customer") or "Unknown")
        if cust in ("", "Unknown", "None"):
            continue
        g = by_cust[cust]
        g["customer"] = cust
        g["record_type"] = "customer_summary"
        g["order_count"] += 1
        g["net_sales"] += _f(r.get("NetSales"))
        g["gross_margin"] += _f(r.get("GrossMargin"))
        g["total_qty"] += _f(r.get("Total Qty"))

    cust_rows = sorted(by_cust.values(), key=lambda x: x["net_sales"], reverse=True)[:100]
    for g in cust_rows:
        g["gross_margin_pct"] = _pct(g["gross_margin"], g["net_sales"])
    _save([dict(g) for g in cust_rows], out / "header_top_customers.json")

    # 5. Discount usage
    by_discount: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in records:
        desc = str(r.get("discountDescription") or "none")
        if not desc or desc.lower() in ("none", "null", ""):
            desc = "no_discount"
        g = by_discount[desc]
        g["discount_code"] = desc
        g["record_type"] = "discount_summary"
        g["order_count"] += 1
        g["total_discount_amount"] += _f(r.get("DiscountTotal"))
        g["net_sales"] += _f(r.get("NetSales"))

    discount_rows = sorted(by_discount.values(), key=lambda x: x["order_count"], reverse=True)[:50]
    _save([dict(g) for g in discount_rows], out / "header_by_discount.json")


# ── LinesResults aggregations ─────────────────────────────────────────────────

def aggregate_lines(records: list[dict], out: Path) -> None:
    logger.info("Aggregating %d order lines …", len(records))

    # 1. Overall lines summary
    overall: dict = defaultdict(float)
    overall["record_type"] = "overall_lines_summary"
    overall["total_lines"] = len(records)
    skus: set[str] = set()

    for r in records:
        overall["total_net_sales"] += _f(r.get("NetSales"))
        overall["total_cost"] += _f(r.get("Total Cost"))
        overall["total_material_cost"] += _f(r.get("Material Cost"))
        overall["total_packaging_cost"] += _f(r.get("Packaging Cost"))
        overall["total_labour_cost"] += _f(r.get("Labour Cost"))
        overall["total_distribution_cost"] += _f(r.get("Distribution Cost"))
        overall["total_wastage_cost"] += _f(r.get("Wastage Cost"))
        overall["total_discount"] += _f(r.get("Discount"))
        overall["total_commission"] += _f(r.get("Amazon/ebay Commission"))
        if r.get("styleCode"):
            skus.add(str(r["styleCode"]))

    overall["gross_margin"] = overall["total_net_sales"] - overall["total_cost"]
    overall["gross_margin_pct"] = _pct(overall["gross_margin"], overall["total_net_sales"])
    overall["unique_skus"] = len(skus)
    _save([dict(overall)], out / "lines_overall_summary.json")

    # 2. By SKU / product (top 200 by net sales)
    by_sku: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in records:
        sku = str(r.get("styleCode") or "unknown")
        name = str(r.get("name") or "")
        key = f"{sku}|{name}"
        g = by_sku[key]
        g["style_code"] = sku
        g["product_name"] = name
        g["record_type"] = "sku_summary"
        g["line_count"] += 1
        g["net_sales"] += _f(r.get("NetSales"))
        g["total_cost"] += _f(r.get("Total Cost"))
        g["material_cost"] += _f(r.get("Material Cost"))
        g["packaging_cost"] += _f(r.get("Packaging Cost"))
        g["labour_cost"] += _f(r.get("Labour Cost"))
        g["distribution_cost"] += _f(r.get("Distribution Cost"))
        g["wastage_cost"] += _f(r.get("Wastage Cost"))
        g["discount"] += _f(r.get("Discount"))
        g["commission"] += _f(r.get("Amazon/ebay Commission"))

    sku_rows = sorted(by_sku.values(), key=lambda x: x["net_sales"], reverse=True)[:200]
    for g in sku_rows:
        g["gross_margin"] = g["net_sales"] - g["total_cost"]
        g["gross_margin_pct"] = _pct(g["gross_margin"], g["net_sales"])
    _save([dict(g) for g in sku_rows], out / "lines_by_sku.json")

    # 3. By Month (lines)
    by_month: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in records:
        mo = _month(str(r.get("CreatedDate") or ""))
        g = by_month[mo]
        g["month"] = mo
        g["record_type"] = "lines_monthly_summary"
        g["line_count"] += 1
        g["net_sales"] += _f(r.get("NetSales"))
        g["total_cost"] += _f(r.get("Total Cost"))
        g["discount"] += _f(r.get("Discount"))
        g["commission"] += _f(r.get("Amazon/ebay Commission"))

    month_rows = []
    for mo, g in by_month.items():
        g["gross_margin"] = g["net_sales"] - g["total_cost"]
        g["gross_margin_pct"] = _pct(g["gross_margin"], g["net_sales"])
        month_rows.append(dict(g))
    month_rows.sort(key=lambda x: x["month"])
    _save(month_rows, out / "lines_by_month.json")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    raw_folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/uploads")
    out_folder = Path(sys.argv[2] if len(sys.argv) > 2 else "data/aggregated")
    out_folder.mkdir(parents=True, exist_ok=True)

    header_file = raw_folder / "HeaderResults.json"
    lines_file = raw_folder / "LinesResults.json"

    if header_file.exists():
        logger.info("Loading %s (%.0f MB) …", header_file.name, header_file.stat().st_size / 1e6)
        headers = json.loads(header_file.read_text(encoding="utf-8"))
        aggregate_headers(headers, out_folder)
        del headers  # free memory before loading lines
    else:
        logger.warning("HeaderResults.json not found in %s — skipping.", raw_folder)

    if lines_file.exists():
        logger.info("Loading %s (%.0f MB) …", lines_file.name, lines_file.stat().st_size / 1e6)
        lines = json.loads(lines_file.read_text(encoding="utf-8"))
        aggregate_lines(lines, out_folder)
        del lines
    else:
        logger.warning("LinesResults.json not found in %s — skipping.", raw_folder)

    agg_files = list(out_folder.glob("*.json"))
    total_records = 0
    for f in agg_files:
        data = json.loads(f.read_text())
        total_records += len(data) if isinstance(data, list) else 1

    logger.info(
        "Done. Wrote %d aggregated files with %d total records to %s",
        len(agg_files), total_records, out_folder,
    )
    logger.info("Next step: python scripts/ingest_bi_json.py %s", out_folder)


if __name__ == "__main__":
    main()
