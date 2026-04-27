"""Smoke test for Azure Blob Storage connection.

Run from the project root:
    python test_connection.py
"""

import sys
from datetime import datetime, timezone

from src.azure_client import get_service_client, download_blobs_in_window, list_blobs
from src.config import (
    CONTAINER_ROAD_CLOSURES,
    CONTAINER_TRAIN_MOMENTS,
    CONTAINER_DARWIN_REALTIME,
    CONTAINER_DARWIN_TIMETABLE,
    CONTAINER_RAIL_ROAD_DATA,
    ROAD_DIR,
)

CONTAINERS = [
    CONTAINER_ROAD_CLOSURES,
    CONTAINER_TRAIN_MOMENTS,
    CONTAINER_DARWIN_REALTIME,
    CONTAINER_DARWIN_TIMETABLE,
    CONTAINER_RAIL_ROAD_DATA,
]


def test_connection():
    print("1. Testing connection...")
    client = get_service_client()
    print(f"   Connected to: {client.account_name}\n")


def test_list_containers():
    print("2. Listing containers...")
    client = get_service_client()
    containers = [c.name for c in client.list_containers()]
    print(f"   Found {len(containers)} container(s): {containers}\n")


def test_list_blobs():
    print("3. Listing blobs per container...")
    for name in CONTAINERS:
        try:
            blobs = list_blobs(name)
            preview = blobs[:5]
            suffix = f"... (+{len(blobs) - 5} more)" if len(blobs) > 5 else ""
            print(f"   {name}: {len(blobs)} blob(s)")
            for b in preview:
                print(f"      - {b}")
            if suffix:
                print(f"      {suffix}")
        except Exception as e:
            print(f"   {name}: ERROR - {e}")
    print()


def test_download_sample():
    print("4. Test download (road closures, 24h window)...")
    start = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
    end   = datetime(2026, 4, 10, 23, 59, 59, tzinfo=timezone.utc)

    count = download_blobs_in_window(CONTAINER_ROAD_CLOSURES, ROAD_DIR, start, end)
    if count > 0:
        print(f"   Success - {count} file(s) saved to '{ROAD_DIR}'\n")
    else:
        print(f"   No blobs found in that window (try adjusting the dates)\n")


if __name__ == "__main__":
    try:
        test_connection()
        test_list_containers()
        test_list_blobs()
        test_download_sample()
        print("All tests passed.")
    except Exception as e:
        print(f"\nFailed: {e}", file=sys.stderr)
        sys.exit(1)
