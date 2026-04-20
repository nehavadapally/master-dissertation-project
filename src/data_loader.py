"""Data loading — fetches from Azure with local caching, returns DataFrames.

Each public function checks for local files first, downloads from Azure
only if needed, and returns a clean DataFrame.
"""

import io
import json
import os
import xml.etree.ElementTree as ET

import pandas as pd

from src.azure_client import (
    download_blobs_in_window,
    get_local_files_in_window,
    get_service_client,
    download_blob_by_name
)
from src.config import (
    CONTAINER_DARWIN_TIMETABLE,
    CONTAINER_RAIL_ROAD_DATA,
    CONTAINER_ROAD_CLOSURES,
    CONTAINER_TRAIN_MOMENTS,
    DARWIN_TIMETABLE_DIR,
    RAIL_DIR,
    ROAD_DIR,
    TRAIN_DIR,
)
from src.parsers import load_file, parse_darwin_timetable_files


# ---------------------------------------------------------------------------
# Road closures
# ---------------------------------------------------------------------------

def load_road_closures(start_utc, end_utc):
    """Load planned/unplanned road closure files, fetching from Azure if needed."""
    local = get_local_files_in_window(ROAD_DIR, start_utc, end_utc)

    if not local:
        print("No local files found. Fetching from Azure...")
        download_blobs_in_window(CONTAINER_ROAD_CLOSURES, ROAD_DIR, start_utc, end_utc)
    else:
        print(f"Using {len(local)} local files within the time window.")

    frames = []
    for fname in os.listdir(ROAD_DIR):
        lower = fname.lower()
        if "planned" not in lower and "unplanned" not in lower:
            continue

        df = load_file(os.path.join(ROAD_DIR, fname))
        df["closure_type"] = "unplanned" if "unplanned" in lower else "planned"
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------

def load_stations():
    """Load the GB stations CSV from the local rail directory."""
    
    path = download_blob_by_name(CONTAINER_RAIL_ROAD_DATA, "gb_stations.csv", RAIL_DIR)
    return pd.read_csv(path)


def load_stations_lookup():
    """Download the CORPUS extract and return a normalised TIPLOC lookup DataFrame."""
    client = get_service_client()
    container = client.get_container_client(CONTAINER_RAIL_ROAD_DATA)

    for blob in container.list_blobs():
        if blob.name.lower() == "corpusextract.json":
            os.makedirs(RAIL_DIR, exist_ok=True)
            blob_client = container.get_blob_client(blob.name)
            path = os.path.join(RAIL_DIR, blob.name.replace("/", "_"))

            with open(path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            break

    path = os.path.join(RAIL_DIR, "corpusextract.json")
    raw = pd.read_json(path)

    def _parse(x):
        return json.loads(x) if isinstance(x, str) else x

    lookup = pd.json_normalize(raw["TIPLOCDATA"].apply(_parse))
    lookup.columns = lookup.columns.str.lower()
    return lookup


# ---------------------------------------------------------------------------
# Train moments
# ---------------------------------------------------------------------------

def load_train_moment_files(start_utc, end_utc):
    """Download train-moment blobs if needed and return list of file paths."""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    local = get_local_files_in_window(TRAIN_DIR, start_utc, end_utc)

    if not local:
        print("No local files found. Fetching from Azure...")
        download_blobs_in_window(CONTAINER_TRAIN_MOMENTS, TRAIN_DIR, start_utc, end_utc)
    else:
        print(f"Using {len(local)} local files within the time window.")

    return [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR)]


def parse_train_moments(train_files, stations_lookup_df):
    """Parse train-moment files into a DataFrame with timestamps and station codes.

    Args:
        train_files: list of file paths from load_train_moment_files().
        stations_lookup_df: TIPLOC lookup from load_stations_lookup().

    Returns:
        DataFrame with parsed timestamps and mapped station_code column.
    """
    frames = []
    for fpath in train_files:
        df = load_file(fpath)
        df.columns = df.columns.str.strip().str.lower()

        for col in ("actual_timestamp", "planned_timestamp", "gbtt_timestamp"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: pd.to_datetime(int(x), unit="ms") if pd.notna(x) else None
                )

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    # Map STANOX → 3ALPHA station code
    station_map = dict(zip(stations_lookup_df["stanox"], stations_lookup_df["3alpha"]))
    result["station_code"] = (
        result["loc_stanox"]
        .apply(lambda x: str(int(x)).zfill(5) if pd.notna(x) else None)
        .map(station_map)
    )

    return result


# ---------------------------------------------------------------------------
# Darwin timetable feeds
# ---------------------------------------------------------------------------

def _strip_ns(tag):
    """Remove XML namespace prefix."""
    return tag.split("}")[-1]


def _element_to_dict(elem):
    """Convert an XML element to a namespace-free dict (recursive)."""
    return {
        "tag": _strip_ns(elem.tag),
        "attributes": dict(elem.attrib),
        "text": elem.text.strip() if elem.text else "",
        "children": [_element_to_dict(c) for c in elem],
    }


def load_darwin_timetable():
    """Stream-parse Darwin timetable XMLs from Azure → JSON on disk → DataFrame."""
    client = get_service_client()
    container = client.get_container_client(CONTAINER_DARWIN_TIMETABLE)
    os.makedirs(DARWIN_TIMETABLE_DIR, exist_ok=True)
    print(f"Connected to container: {container.container_name}")

    for blob in container.list_blobs():
        if not blob.name.lower().endswith(".xml"):
            continue

        print(f"Processing: {blob.name}")
        stream = container.get_blob_client(blob.name).download_blob()
        file_like = io.BytesIO(stream.readall())

        journeys = []
        for _, elem in ET.iterparse(file_like, events=("end",)):
            if _strip_ns(elem.tag) == "Journey":
                journeys.append(_element_to_dict(elem))
                elem.clear()

        out_name = os.path.basename(blob.name).replace(".xml", ".json")
        out_path = os.path.join(DARWIN_TIMETABLE_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(journeys, f, indent=2)

    files = [os.path.join(DARWIN_TIMETABLE_DIR, f) for f in os.listdir(DARWIN_TIMETABLE_DIR)]
    return parse_darwin_timetable_files(files)
