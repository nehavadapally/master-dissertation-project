"""Azure Blob Storage client with local file caching."""

import os
import re
from datetime import datetime, timezone

import pandas as pd
from azure.storage.blob import BlobServiceClient

from src.config import AZURE_CONNECTION_STRING

# Filename timestamp patterns
# Pattern A/B: planned_20260411_050919.csv
_PAT_UNDERSCORE   = re.compile(r"(\d{8})_(\d{6})")
# Pattern C:   PPTimetable_20260410020459_v8.json
_PAT_NO_UNDERSCORE = re.compile(r"(\d{14})")


def get_service_client() -> BlobServiceClient:
    """Create and return an Azure BlobServiceClient."""
    return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)


def _parse_blob_timestamp(filename: str) -> datetime:
    """Extract a UTC datetime from a blob filename.

    Supports two formats:
      - <name>_YYYYMMDD_HHMMSS.<ext>  (road / train files)
      - <name>_YYYYMMDDHHMMSS_<ver>.<ext>  (Darwin timetable files)

    Raises ValueError if neither pattern matches.
    """
    base = os.path.basename(filename)
    match = _PAT_UNDERSCORE.search(base)
    if match:
        date_part, time_part = match.groups()
        blob_dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
    else:
        match = _PAT_NO_UNDERSCORE.search(base)
        if not match:
            raise ValueError(f"No timestamp found in filename: {filename}")
        blob_dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return blob_dt.replace(tzinfo=timezone.utc)


def download_blobs_in_window(
    container_name: str, local_dir: str, start_utc: datetime, end_utc: datetime
) -> int:
    """Download all blobs whose filename timestamp falls within [start_utc, end_utc].

    Returns the number of blobs downloaded.
    """
    container = get_service_client().get_container_client(container_name)
    os.makedirs(local_dir, exist_ok=True)
    print(f"Connected to container: {container_name}")

    count = 0
    for blob in container.list_blobs():
        try:
            blob_dt = _parse_blob_timestamp(blob.name)
        except ValueError as e:
            print(f"Skipping blob '{blob.name}': {e}")
            continue

        if start_utc <= blob_dt <= end_utc:
            safe_name = blob.name.replace("/", "_")
            path = os.path.join(local_dir, safe_name)
            with open(path, "wb") as f:
                f.write(container.get_blob_client(blob.name).download_blob().readall())
            count += 1

    print(f"Downloaded {count} blob(s) to '{local_dir}'")
    return count


def download_blob_by_name(
    container_name: str, blob_name: str, local_dir: str
) -> str:
    """Download a single blob by name, using local cache if already present.

    Returns the local file path.
    """
    os.makedirs(local_dir, exist_ok=True)
    safe_name = blob_name.replace("/", "_")
    local_path = os.path.join(local_dir, safe_name.lower())

    if not os.path.exists(local_path):
        container = get_service_client().get_container_client(container_name)
        with open(local_path, "wb") as f:
            f.write(container.get_blob_client(blob_name).download_blob().readall())

    return local_path


def get_local_files_in_window(
    dir_path: str, start_utc: datetime, end_utc: datetime
) -> list[str]:
    """Return paths of local files whose filename timestamp falls within the UTC window."""
    if not os.path.isdir(dir_path):
        return []

    matching = []
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            file_dt = _parse_blob_timestamp(fname)
        except ValueError:
            print(f"Could not parse filename: {fname}")
            continue

        if start_utc <= file_dt <= end_utc:
            matching.append(fpath)

    return matching