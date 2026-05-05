"""Data loading - fetches from Azure with local caching, returns DataFrames.

Each public function checks for local files first, downloads from Azure
only if needed, and returns a clean DataFrame.
"""

from importlib.resources import path
import io
import json
import os
import xml.etree.ElementTree as ET
import requests

from numpy import dtype
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
    HSP_BASE_URL,
    HSP_SERVICE_DETAILS_HEADERS 
)
from src.parsers import load_file, parse_darwin_timetable_files, parse_service_details
# ---------------------------------------------------------------------------
# Road Network Data
# ---------------------------------------------------------------------------

def load_traffic_count_aadf():
    """Load the AADF traffic count data, fetching from Azure if needed."""
    file_path = download_blob_by_name(
        CONTAINER_RAIL_ROAD_DATA,
        "dft_traffic_counts_aadf.csv",
        ROAD_DIR
    )
    return pd.read_csv(file_path)

def laod_traffic_count_raw():
    """Load the raw traffic count data, fetching from Azure if needed."""
    file_path = download_blob_by_name(
        CONTAINER_RAIL_ROAD_DATA,
        "dft_traffic_counts_raw.csv",
        ROAD_DIR
    )
    return pd.read_csv(file_path)
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
    """Load GB stations CSV and corpus extract JSON, merge on TLC."""

    # Download or load cached files
    file_path = download_blob_by_name(
        CONTAINER_RAIL_ROAD_DATA,
        "gb_stations.csv",
        RAIL_DIR
    )
    return pd.read_csv(file_path)




def load_stations_lookup():
    """Download the CORPUS extract and return a normalised TIPLOC lookup DataFrame."""
    # Download or load cached files
    file_path = download_blob_by_name(
        CONTAINER_RAIL_ROAD_DATA,
        "CORPUSExtract.json",
        RAIL_DIR
    )
    
    raw = pd.read_json(file_path)

    def _parse(x):
        return json.loads(x) if isinstance(x, str) else x

    lookup = pd.json_normalize(raw["TIPLOCDATA"].apply(_parse))
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


def parse_train_moments(train_files, stations_df):
    """Parse train-moment files into a DataFrame with timestamps and station codes.

    Args:
        train_files: list of file paths from load_train_moment_files().
        stations_df: DataFrame of stations from load_stations().

    Returns:
        DataFrame with parsed timestamps and mapped station_code column.
    """
    frames = []
    for fpath in train_files:
        df = load_file(fpath) 
        if len(df) > 0:
            df.columns = df.columns.str.strip().str.lower()

            for col in ("actual_timestamp", "planned_timestamp", "gbtt_timestamp"):
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: pd.to_datetime(int(x), unit="ms") if pd.notna(x) else None
                    )
    
            frames.append(df)
        else:
            continue

    result = pd.concat(frames, ignore_index=True)

    # Map STANOX → 3ALPHA station code
    station_map = dict(zip(stations_df["stanox"], stations_df["tlc"]))
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


def load_darwin_timetable(start_utc, end_utc):
    """
    Load Darwin timetable files:
    1. Check local directory first (filter by filename datetime)
    2. Download missing files from Azure (only those in the window)
    3. Parse timetable JSON files into a DataFrame
    """

    os.makedirs(DARWIN_TIMETABLE_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1 - Get local files in the time window
    # ---------------------------------------------------------
    local_files = get_local_files_in_window(DARWIN_TIMETABLE_DIR, start_utc, end_utc)

    # If we already have all files locally, skip Azure
    if local_files:
        return parse_darwin_timetable_files(local_files)

    # ---------------------------------------------------------
    # STEP 2 - Otherwise fetch from Azure
    # ---------------------------------------------------------
    client = get_service_client()
    container = client.get_container_client(CONTAINER_DARWIN_TIMETABLE)
    print(f"Connected to container: {container.container_name}")

    downloaded_files = []

    for blob in container.list_blobs():
        if not blob.name.lower().endswith(".xml"):
            continue

        # Extract datetime from filename
        try:
            base = os.path.splitext(os.path.basename(blob.name))[0]
            digits = "".join(c for c in base if c.isdigit())[:14]
            file_dt = pd.to_datetime(digits, format="%Y%m%d%H%M%S", utc=True)
        except Exception as e:
            print(f"Could not parse timetable filename: {blob.name}, Exception: {e}")
            continue

        # Check if blob is inside the requested window
        if not (start_utc <= file_dt <= end_utc):
            continue

        print(f"Downloading + parsing: {blob.name}")

        # Download XML
        stream = container.get_blob_client(blob.name).download_blob()
        file_like = io.BytesIO(stream.readall())

        # Parse XML → JSON
        journeys = []
        for _, elem in ET.iterparse(file_like, events=("end",)):
            if _strip_ns(elem.tag) == "Journey":
                journeys.append(_element_to_dict(elem))
                elem.clear()

        # Save JSON locally
        out_name = os.path.basename(blob.name).replace(".xml", ".json")
        out_path = os.path.join(DARWIN_TIMETABLE_DIR, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(journeys, f, indent=2)

        downloaded_files.append(out_path)

    # ---------------------------------------------------------
    # STEP 3 - Parse all timetable JSON files
    # ---------------------------------------------------------
    return parse_darwin_timetable_files(downloaded_files)

def get_service_details(timetable_df):
    url = f"{HSP_BASE_URL}/serviceDetails"
    print("get service etails")
    # print(timetable_df)
    df = timetable_df[1:len(timetable_df)]
    print(df)
    for row in df:
        print(row)
        payload = {"rid": row['rid']}

        try:
            response = requests.post(url, json=payload, headers=HSP_SERVICE_DETAILS_HEADERS)

            if response.status_code != 200:
                print(f"Error for RID {row['rid']}:", response.text)
                return None

            if response.json():
                return parse_service_details(response.json())
            break
        except Exception as e:
            print(f"Exception for RID {row['rid']}:", e)
            return None