import sqlite3
import os
import logging
from connectors.gdrive import list_drive_files, download_file_to_disk
from processing.chunker import process_file
from search.vector_store import save_chunks_to_index, invalidate_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — validate at import time so errors surface immediately
# ---------------------------------------------------------------------------
DB_FILE = "sync_state.db"


def init_db():
    """Creates the SQLite tracking table if it doesn't already exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS synced_files (
            drive_file_id TEXT PRIMARY KEY,
            file_name     TEXT,
            modified_time TEXT
        )
    """)
    conn.commit()
    return conn


def run_incremental_sync() -> dict:
    """
    Scans the configured Google Drive folder and syncs only new or updated PDFs.
    Returns a stats dict: {new, updated, skipped, errors}.
    """
    target_folder_id = os.getenv("GDRIVE_FOLDER_ID")
    if not target_folder_id:
        raise EnvironmentError("GDRIVE_FOLDER_ID is not set. Add it to your .env file.")

    conn = init_db()
    cursor = conn.cursor()

    drive_files = list_drive_files(target_folder_id)
    stats = {"new": 0, "updated": 0, "skipped": 0, "errors": []}

    ALLOWED_MIMES = [
        "application/pdf", 
        "text/plain", 
        "application/vnd.google-apps.document"
    ]

    for file in drive_files:
        mime_type = file["mimeType"]
        if mime_type not in ALLOWED_MIMES:
            logger.debug("Skipping unsupported file: %s (%s)", file["name"], mime_type)
            continue

        file_id = file["id"]
        file_name = file["name"]
        drive_mod_time = file["modifiedTime"]

        try:
            cursor.execute(
                "SELECT modified_time FROM synced_files WHERE drive_file_id=?",
                (file_id,),
            )
            row = cursor.fetchone()

            # ── SCENARIO A: New file ─────────────────────────────────────────
            if row is None:
                logger.info("Syncing NEW file: %s", file_name)
                local_path = download_file_to_disk(file_id, file_name, mime_type)
                chunks = process_file(local_path, file_id, file_name)
                save_chunks_to_index(chunks)
                cursor.execute(
                    "INSERT INTO synced_files VALUES (?, ?, ?)",
                    (file_id, file_name, drive_mod_time),
                )
                stats["new"] += 1

            # ── SCENARIO B: Updated file ─────────────────────────────────────
            elif row[0] < drive_mod_time:
                logger.info("Syncing UPDATED file: %s", file_name)
                local_path = download_file_to_disk(file_id, file_name, mime_type)
                chunks = process_file(local_path, file_id, file_name)
                save_chunks_to_index(chunks)
                cursor.execute(
                    "UPDATE synced_files SET modified_time=? WHERE drive_file_id=?",
                    (drive_mod_time, file_id),
                )
                stats["updated"] += 1

            # ── SCENARIO C: Unchanged file ───────────────────────────────────
            else:
                logger.debug("Skipping UNCHANGED file: %s", file_name)
                stats["skipped"] += 1

        except Exception as exc:
            # A single bad file must NOT crash the whole batch
            error_msg = f"Failed to process '{file_name}': {exc}"
            logger.error(error_msg, exc_info=True)
            stats["errors"].append(error_msg)

    conn.commit()
    conn.close()

    # Bust the in-memory FAISS cache so /ask picks up the new data immediately
    if stats["new"] > 0 or stats["updated"] > 0:
        invalidate_cache()

    logger.info("Sync complete — %s", stats)
    return stats