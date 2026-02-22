#!/usr/bin/env python3
"""Ingest JSON files from a folder into the BI vector store.

Runs before the main app starts (called from the Docker entrypoint).
Only processes .json files; other file types are ignored.

IMPORTANT: Point this at the AGGREGATED folder (data/aggregated/), not the
raw uploads folder. Run prepare_bi_data.py first to produce the aggregated
summaries from raw HeaderResults.json / LinesResults.json files.

Usage:
    python scripts/ingest_bi_json.py [folder]

    folder: path to scan for .json files (default: data/aggregated)
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ingest_json_folder(folder: Path) -> int:
    """Ingest all .json files in *folder* using the internal IngestService.

    Returns the number of files ingested.
    """
    # Import here so the settings system is initialised before DI is wired up.
    from private_gpt.di import global_injector
    from private_gpt.server.ingest.ingest_service import IngestService

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        logger.info("No .json files found in %s — skipping ingestion.", folder)
        return 0

    logger.info("Found %d .json file(s) to ingest from %s", len(json_files), folder)
    ingest_service = global_injector.get(IngestService)

    ingested = 0
    for json_file in json_files:
        try:
            logger.info("Ingesting %s …", json_file.name)
            ingest_service.ingest_file(json_file.name, json_file)
            logger.info("  ✓ %s ingested successfully.", json_file.name)
            ingested += 1
        except Exception:
            logger.exception("  ✗ Failed to ingest %s — skipping.", json_file.name)

    logger.info("Done. %d/%d file(s) ingested.", ingested, len(json_files))
    return ingested


if __name__ == "__main__":
    # Default to aggregated summaries, not raw uploads.
    # Run prepare_bi_data.py on raw data first.
    folder_arg = sys.argv[1] if len(sys.argv) > 1 else "data/aggregated"
    uploads_folder = Path(folder_arg)

    if not uploads_folder.exists():
        logger.info("Folder %s does not exist — creating it.", uploads_folder)
        uploads_folder.mkdir(parents=True, exist_ok=True)

    ingest_json_folder(uploads_folder)
