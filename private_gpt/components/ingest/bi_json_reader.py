"""BI-aware JSON reader.

Produces one Document per JSON record so each row is independently
retrievable during RAG. Supports:
  - Array of objects  (most common BI format — pre-aggregated summaries)
  - Single object     (treated as one record)
  - Nested objects    (flattened to "parent.child: value" text)

For very large raw JSON files (>10 MB / >5000 records) use the
prepare_bi_data.py aggregation script first, then ingest the aggregated
output. This reader will warn and truncate if a raw file is too large.
"""
import json
import logging
from pathlib import Path
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

# Warn when a single JSON file has more records than this.
# Aggregated summary files should always be well below this limit.
_RECORD_LIMIT = 5_000


def _flatten(obj: Any, prefix: str = "") -> dict[str, str]:
    """Recursively flatten a nested dict/list into {dotted.key: str_value}."""
    items: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            items.update(_flatten(v, full_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            full_key = f"{prefix}[{i}]"
            items.update(_flatten(v, full_key))
    else:
        # Round floats to 4 dp so embedding text is compact and stable
        if isinstance(obj, float):
            items[prefix] = f"{obj:.4f}"
        else:
            items[prefix] = str(obj) if obj is not None else ""
    return items


def _record_to_text(record: Any, index: int) -> str:
    """Convert a single JSON record to a readable key: value block."""
    flat = _flatten(record)
    lines = [f"Record {index}:"] + [f"  {k}: {v}" for k, v in flat.items()]
    return "\n".join(lines)


class BIJsonReader(BaseReader):
    """Load a JSON file and produce one Document per top-level record.

    Designed for business intelligence datasets where the file is a
    pre-aggregated array of summary objects (output of prepare_bi_data.py).
    """

    def load_data(self, file: Path, extra_info: dict | None = None) -> list[Document]:  # type: ignore[override]
        file_mb = file.stat().st_size / 1e6
        if file_mb > 10:
            logger.warning(
                "BIJsonReader: %s is %.0f MB. Run prepare_bi_data.py first to "
                "aggregate raw data before ingesting. Attempting to load anyway …",
                file.name, file_mb,
            )

        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("BIJsonReader could not parse %s: %s", file, exc)
            return []

        records: list[Any] = raw if isinstance(raw, list) else [raw]
        total = len(records)

        if total > _RECORD_LIMIT:
            logger.warning(
                "BIJsonReader: %s has %d records (limit %d). "
                "Only ingesting the first %d. Run prepare_bi_data.py to aggregate first.",
                file.name, total, _RECORD_LIMIT, _RECORD_LIMIT,
            )
            records = records[:_RECORD_LIMIT]

        logger.debug("BIJsonReader: ingesting %d/%d records from %s", len(records), total, file.name)

        documents: list[Document] = []
        for i, record in enumerate(records):
            text = _record_to_text(record, i)
            metadata: dict[str, Any] = {
                "source_file": file.name,
                "record_index": i,
                "total_records": total,
            }
            if extra_info:
                metadata.update(extra_info)
            documents.append(Document(text=text, metadata=metadata))

        return documents
