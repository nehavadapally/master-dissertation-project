"""Azure Blob Storage client with local file caching."""

import os

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

    count = 0
    for blob in container.list_blobs():
        if start_utc <= blob.last_modified <= end_utc:
            blob_client = container.get_blob_client(blob.name)
            safe_name = blob.name.replace("/", "_")
            path = os.path.join(local_dir, safe_name)

            with open(path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            count += 1

    print(f"Downloaded {count} blob(s) to '{local_dir}'")
    return count


def download_blob_by_name(container_name, blob_name, local_dir):
    """Download a single blob by exact name."""
    container = get_service_client().get_container_client(container_name)
    os.makedirs(local_dir, exist_ok=True)

    blob_client = container.get_blob_client(blob_name)
    safe_name = blob_name.replace("/", "_")
    path = os.path.join(local_dir, safe_name)

    with open(path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    return path


def get_local_files_in_window(dir_path, start_utc, end_utc):
    """Return local file paths whose last-modified time falls within the UTC window."""
    if not os.path.isdir(dir_path):
        return []

    matching = []
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue

        mtime = pd.to_datetime(os.path.getmtime(fpath), unit="s", utc=True)
        if start_utc <= mtime <= end_utc:
            matching.append(fpath)

    return matching


def list_blobs(container_name):
    """List all blob names in a container."""
    container = get_service_client().get_container_client(container_name)
    return [blob.name for blob in container.list_blobs()]
