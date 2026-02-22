#!/usr/bin/env python3
"""Pre-aggregate raw BI JSON files into compact summary documents for RAG.

Randomly samples N orders (OriginalReference) from HeaderResults.json,
pulls the matching line items from LinesResults.json, then computes
aggregations over that sample. This keeps the vector store small and
embedding fast while remaining representative.

Usage:
    python scripts/prepare_bi_data.py [raw_folder] [out_folder] [sample_size]

    raw_folder   default: data/uploads
    out_folder   default: data/aggregated
    sample_size  default: 25
"""
import json
import logging
import random
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
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _month(date_str: str) -> str:
    try:
        return datetime.fromisoformat(date_str).strftime("%Y-%m")
    except Exception:
        return "unknown"


def _pct(numerator: float, denominator: float) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def _save(data: list | dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("  Wrote %s (%d records)", path.name, len(data) if isinstance(data, list) else 1)


# ── sampling ──────────────────────────────────────────────────────────────────

def sample_references(all_headers: list[dict], n: int) -> set[str]:
    """Pick n unique OriginalReference values at random."""
    refs = [str(r["OriginalReference"]) for r in all_headers if r.get("OriginalReference")]
    unique = list(dict.fromkeys(refs))   # deduplicate, preserve order
    k = min(n, len(unique))
    chosen = set(random.sample(unique, k))
    logger.info("Sampled %d / %d unique OriginalReferences.", k, len(unique))
    return chosen


# ── HeaderResults aggregations ────────────────────────────────────────────────

def aggregate_headers(records: list[dict], refs: set[str], out: Path) -> None:
    logger.info("Aggregating %d sampled order headers …", len(records))

    # 0. Raw order records (one doc per order — for order-level lookup)
    raw_rows = []
    for r in records:
        raw_rows.append({
            "record_type": "order_detail",
            "original_reference": r.get("OriginalReference"),
            "sales_channel": r.get("Sales Channel"),
            "created_date": r.get("CreatedDate"),
            "customer": r.get("Customer"),
            "net_sales": _f(r.get("NetSales")),
            "total_sales_inc_vat": _f(r.get("TotalSales")),
            "total_qty": _f(r.get("Total Qty")),
            "gross_margin": _f(r.get("GrossMargin")),
            "gross_margin_pct": _pct(_f(r.get("GrossMargin")), _f(r.get("NetSales"))),
            "discount_code": r.get("discountDescription") or "none",
            "discount_amount": _f(r.get("DiscountTotal")),
            "distribution_cost": _f(r.get("Distribution Cost")),
            "material_cost": _f(r.get("Material Cost")),
            "commission": _f(r.get("Amazon/ebay Commission")),
        })
    _save(raw_rows, out / "header_orders.json")

    # 1. Overall summary
    overall: dict = defaultdict(float)
    overall["record_type"] = "overall_order_summary"
    overall["total_orders"] = len(records)
    overall["sampled_references"] = len(refs)
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

    # 4. By Customer
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

    cust_rows = sorted(by_cust.values(), key=lambda x: x["net_sales"], reverse=True)
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

    discount_rows = sorted(by_discount.values(), key=lambda x: x["order_count"], reverse=True)
    _save([dict(g) for g in discount_rows], out / "header_by_discount.json")


# ── LinesResults aggregations ─────────────────────────────────────────────────

def aggregate_lines(records: list[dict], out: Path) -> None:
    logger.info("Aggregating %d sampled order lines …", len(records))

    # 0. Raw line records (one doc per line — for product-level lookup)
    raw_rows = []
    for r in records:
        raw_rows.append({
            "record_type": "line_detail",
            "original_reference": r.get("OriginalReference"),
            "created_date": r.get("CreatedDate"),
            "style_code": r.get("styleCode"),
            "product_name": r.get("name"),
            "net_sales": _f(r.get("NetSales")),
            "total_cost": _f(r.get("Total Cost")),
            "gross_margin": _f(r.get("NetSales")) - _f(r.get("Total Cost")),
            "gross_margin_pct": _pct(
                _f(r.get("NetSales")) - _f(r.get("Total Cost")),
                _f(r.get("NetSales")),
            ),
            "discount": _f(r.get("Discount")),
            "commission": _f(r.get("Amazon/ebay Commission")),
            "material_cost": _f(r.get("Material Cost")),
            "distribution_cost": _f(r.get("Distribution Cost")),
        })
    _save(raw_rows, out / "lines_orders.json")

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

    # 2. By SKU / product
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

    sku_rows = sorted(by_sku.values(), key=lambda x: x["net_sales"], reverse=True)
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
    sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    out_folder.mkdir(parents=True, exist_ok=True)

    header_file = raw_folder / "HeaderResults.json"
    lines_file = raw_folder / "LinesResults.json"

    if not header_file.exists():
        logger.error("HeaderResults.json not found in %s — cannot continue.", raw_folder)
        sys.exit(1)

    # ── 1. Load headers and sample N OriginalReferences ──────────────────────
    logger.info("Loading %s (%.0f MB) …", header_file.name, header_file.stat().st_size / 1e6)
    all_headers: list[dict] = json.loads(header_file.read_text(encoding="utf-8"))

    sampled_refs = sample_references(all_headers, sample_size)

    sampled_headers = [r for r in all_headers if str(r.get("OriginalReference")) in sampled_refs]
    logger.info(
        "Filtered to %d header records matching %d sampled references.",
        len(sampled_headers), len(sampled_refs),
    )
    del all_headers  # free memory

    # ── 2. Save the sampled reference list for traceability ──────────────────
    _save(
        sorted(sampled_refs),
        out_folder / "sampled_references.json",
    )

    # ── 3. Aggregate headers ──────────────────────────────────────────────────
    aggregate_headers(sampled_headers, sampled_refs, out_folder)
    del sampled_headers

    # ── 4. Load lines and filter to sampled references ────────────────────────
    if lines_file.exists():
        logger.info("Loading %s (%.0f MB) …", lines_file.name, lines_file.stat().st_size / 1e6)
        all_lines: list[dict] = json.loads(lines_file.read_text(encoding="utf-8"))

        sampled_lines = [r for r in all_lines if str(r.get("OriginalReference")) in sampled_refs]
        logger.info(
            "Filtered to %d line records matching sampled references.",
            len(sampled_lines),
        )
        del all_lines

        aggregate_lines(sampled_lines, out_folder)
        del sampled_lines
    else:
        logger.warning("LinesResults.json not found in %s — skipping lines.", raw_folder)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    agg_files = list(out_folder.glob("*.json"))
    total_records = sum(
        len(json.loads(f.read_text())) if isinstance(json.loads(f.read_text()), list) else 1
        for f in agg_files
    )
    logger.info(
        "Done. %d aggregated files, %d total records → %s",
        len(agg_files), total_records, out_folder,
    )
    logger.info("Next step: python scripts/ingest_bi_json.py %s", out_folder)


if __name__ == "__main__":
    main()
