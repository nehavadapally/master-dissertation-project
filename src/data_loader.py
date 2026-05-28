"""Data loading - fetches from Azure with local caching, returns DataFrames.

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
    download_blob_by_name,
    get_local_files_in_window,
    get_service_client,
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
from src.parsers import load_file, parse_darwin_timetable_files_parallel


# ---------------------------------------------------------------------------
# Road closures
# ---------------------------------------------------------------------------

def load_road_closures(start_utc, end_utc) -> pd.DataFrame:
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

def load_stations() -> pd.DataFrame:
    """Load GB stations CSV from Azure (cached locally after first download)."""
    file_path = download_blob_by_name(
        CONTAINER_RAIL_ROAD_DATA, "gb_stations.csv", RAIL_DIR
    )
    return pd.read_csv(file_path)


def load_stations_lookup() -> pd.DataFrame:
    """Load CORPUS extract and return a normalised TIPLOC lookup DataFrame."""
    file_path = download_blob_by_name(
        CONTAINER_RAIL_ROAD_DATA, "CORPUSExtract.json", RAIL_DIR
    )
    raw = pd.read_json(file_path)

    def _parse(x):
        return json.loads(x) if isinstance(x, str) else x

    return pd.json_normalize(raw["TIPLOCDATA"].apply(_parse))


# ---------------------------------------------------------------------------
# Train moments
# ---------------------------------------------------------------------------

def load_train_moment_files(start_utc, end_utc) -> list[str]:
    """Download train-moment blobs if needed and return list of local file paths."""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    local = get_local_files_in_window(TRAIN_DIR, start_utc, end_utc)

    if not local:
        print("No local files found. Fetching from Azure...")
        download_blobs_in_window(CONTAINER_TRAIN_MOMENTS, TRAIN_DIR, start_utc, end_utc)
    else:
        print(f"Using {len(local)} local files within the time window.")

    return [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR)]


def parse_train_moments(train_files: list[str], stations_df: pd.DataFrame) -> pd.DataFrame:
    """Parse train-moment files into a DataFrame with timestamps and station codes.

    Args:
        train_files: list of file paths from load_train_moment_files().
        stations_df: stations reference DataFrame from load_stations().

    Returns:
        DataFrame with parsed timestamps and mapped station_code / station_name columns.
    """
    frames = []
    for fpath in train_files:
        df = load_file(fpath)
        if df.empty:
            continue
        df.columns = df.columns.str.strip().str.lower()
        for col in ("actual_timestamp", "planned_timestamp", "gbtt_timestamp"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    # Map STANOX → TLC (3-letter code) and station name
    stanox_col = result["loc_stanox"].apply(
        lambda x: str(int(x)).zfill(5) if pd.notna(x) else None
    )
    result["station_code"] = stanox_col.map(
        dict(zip(stations_df["stanox"], stations_df["tlc"]))
    )
    result["station_name"] = stanox_col.map(
        dict(zip(stations_df["stanox"], stations_df["station"]))
    )
    return result


# ---------------------------------------------------------------------------
# Darwin timetable
# ---------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from a tag string."""
    return tag.split("}")[-1]


def _element_to_dict(elem) -> dict:
    """Convert an XML element to a namespace-free dict (recursive)."""
    return {
        "tag":        _strip_ns(elem.tag),
        "attributes": dict(elem.attrib),
        "text":       elem.text.strip() if elem.text else "",
        "children":   [_element_to_dict(c) for c in elem],
    }


def load_darwin_timetable(start_utc, end_utc) -> pd.DataFrame:
    """Load Darwin timetable files, using local cache where available.

    Steps:
      1. Check local DARWIN_TIMETABLE_DIR for files within the window.
      2. For any missing files, download XML from Azure, parse to JSON and save locally.
      3. Parse all local JSON files in parallel and return a flat DataFrame.
    """
    os.makedirs(DARWIN_TIMETABLE_DIR, exist_ok=True)

    local_files = get_local_files_in_window(DARWIN_TIMETABLE_DIR, start_utc, end_utc)
    if local_files:
        return parse_darwin_timetable_files_parallel(local_files)

    # Fetch from Azure
    container = get_service_client().get_container_client(CONTAINER_DARWIN_TIMETABLE)
    print(f"Connected to container: {container.container_name}")

    downloaded_files = []
    for blob in container.list_blobs():
        if not blob.name.lower().endswith(".xml"):
            continue

        try:
            base   = os.path.splitext(os.path.basename(blob.name))[0]
            digits = "".join(c for c in base if c.isdigit())[:14]
            file_dt = pd.to_datetime(digits, format="%Y%m%d%H%M%S", utc=True)
        except Exception as e:
            print(f"Could not parse timetable filename: {blob.name} - {e}")
            continue

        if not (start_utc <= file_dt <= end_utc):
            continue

        print(f"Downloading: {blob.name}")
        stream    = container.get_blob_client(blob.name).download_blob()
        file_like = io.BytesIO(stream.readall())

        # Stream-parse XML → list of journey dicts
        journeys = []
        for _, elem in ET.iterparse(file_like, events=("end",)):
            if _strip_ns(elem.tag) == "Journey":
                journeys.append(_element_to_dict(elem))
                elem.clear()

        out_path = os.path.join(
            DARWIN_TIMETABLE_DIR,
            os.path.basename(blob.name).replace(".xml", ".json")
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(journeys, f, indent=2)

        downloaded_files.append(out_path)

    return parse_darwin_timetable_files_parallel(downloaded_files)