"""Azure Blob Storage client with local file caching."""

import os
from string import digits
import datetime
from datetime import datetime, timezone
import re
import pandas as pd
from azure.storage.blob import BlobServiceClient

from src.config import AZURE_CONNECTION_STRING


def get_service_client():
    """Create and return an Azure BlobServiceClient."""
    return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)


def download_blobs_in_window(container_name, local_dir, start_utc, end_utc):
    """Download all blobs modified within [start_utc, end_utc] to local_dir."""
    container = get_service_client().get_container_client(container_name)
    os.makedirs(local_dir, exist_ok=True)
    print(f"Connected to container: {container_name}")
    # Pattern A/B: planned_20260411_050919.csv
    pattern_with_underscore = re.compile(r"(\d{8})_(\d{6})")

    # Pattern C: PPTimetable_20260410020459_v8.json
    pattern_no_underscore = re.compile(r"(\d{14})")

    count = 0
    for blob in container.list_blobs():
        base = os.path.basename(blob.name)

        try:
            
            match = pattern_with_underscore.search(base)
            if match:
                date_part, time_part = match.groups()
                timestamp_str = f"{date_part}_{time_part}"
                blob_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

            else:
                
                match = pattern_no_underscore.search(base)
                if not match:
                    raise ValueError("No timestamp found in filename")

                timestamp_str = match.group(1)  
                blob_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")

            # Make timezone-aware
            blob_dt = blob_dt.replace(tzinfo=timezone.utc)

        except Exception as e:
            print(f"Skipping blob '{blob.name}' (cannot parse timestamp): {e}")
            continue

        if start_utc <= blob_dt <= end_utc:
            blob_client = container.get_blob_client(blob.name)
            safe_name = blob.name.replace("/", "_")
            path = os.path.join(local_dir, safe_name)

            with open(path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            count += 1

    print(f"Downloaded {count} blob(s) to '{local_dir}'")
    return count


def download_blob_by_name(container_name, blob_name, local_dir):
    """
    Accepts a list of blob names.
    Checks local_dir for existing files.
    Downloads only missing blobs from Azure.
    Returns list of local file paths.
    """
    container = get_service_client().get_container_client(container_name)
    os.makedirs(local_dir, exist_ok=True)

    safe_name = blob_name.replace("/", "_")
    local_path = os.path.join(local_dir, safe_name.lower())
    # If file already exists locally → skip download
    if not os.path.exists(local_path):
        blob_client = container.get_blob_client(blob_name)
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

    return local_path

def get_local_files_in_window(dir_path, start_utc, end_utc):
    """Return local file paths whose filename datetime falls within the UTC window."""

    if not os.path.isdir(dir_path):
        return []

    matching = []
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue

        file_dt = None

        # ---------------------------------------------------
        # CASE 1: Darwin Timetable files
        # ---------------------------------------------------
        if "data\darwin_timetable" in dir_path.lower():
            try:
                base = os.path.splitext(fname)[0]
                digits = "".join(c for c in base if c.isdigit())[:14]
                file_dt = pd.to_datetime(digits, format="%Y%m%d%H%M%S", utc=True)

            except Exception as e:
                print(f"Could not parse timetable filename: {fname}")
                print(f"Error: {e}")
                continue

        # ---------------------------------------------------
        # CASE 2: Train/Road/Rail data files
        # ---------------------------------------------------
        else:
            try:
                base = os.path.splitext(fname)[0]
                parts = base.split("_")
                date_str = parts[-2]
                time_str = parts[-1]
                file_dt = pd.to_datetime(date_str + time_str,
                                         format="%Y%m%d%H%M%S",
                                         utc=True)

            except Exception:
                print(f"Could not parse filename: {fname}")
                continue

        # ---------------------------------------------------
        # Check if file datetime is inside the window
        # ---------------------------------------------------
        if start_utc <= file_dt <= end_utc:
            matching.append(fpath)
    return matching

def list_blobs(container_name):
    """List all blob names in a container."""
    container = get_service_client().get_container_client(container_name)
    return [blob.name for blob in container.list_blobs()]
