"""File parsing utilities - CSV/JSON loader and Darwin timetable parser."""

import json
from multiprocessing import Pool, cpu_count

import pandas as pd


def load_file(path: str) -> pd.DataFrame:
    """Load a CSV or newline-delimited JSON file into a DataFrame.

    Returns an empty DataFrame for empty CSV files rather than raising.
    """
    if path.endswith(".csv"):
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty CSV file: {path}")
            return pd.DataFrame()
    return pd.read_json(path, lines=True)


def parse_darwin_timetable_files_parallel(timetable_files: list[str]) -> pd.DataFrame:
    """Parse one or more JSON timetable files in parallel and return a flat DataFrame."""
    with Pool(cpu_count()) as pool:
        dfs = pool.map(_parse_one_file, timetable_files)
    return pd.concat(dfs, ignore_index=True)


def _parse_one_file(file_path: str) -> pd.DataFrame:
    """Parse a single JSON timetable file into a flat DataFrame of stop rows."""
    with open(file_path, "r") as f:
        data = json.load(f)

    journeys = data if isinstance(data, list) else [data]
    print(f"Processing {file_path} - {len(journeys):,} journeys")

    rows = []
    for journey in journeys:
        rows.extend(_journey_to_rows(journey))
    return pd.DataFrame(rows)


def _journey_to_rows(journey: dict) -> list[dict]:
    """Flatten a single journey dict into one row per stop."""
    attrs    = journey.get("attributes", {})
    rid      = attrs.get("rid")
    uid      = attrs.get("uid")
    train_id = attrs.get("trainId")
    ssd      = attrs.get("ssd")
    ssd_date = pd.to_datetime(ssd).date()
    toc      = attrs.get("toc")

    rows = []
    for child in journey.get("children", []):
        stop = child.get("attributes", {})
        rows.append({
            "rid":       rid,
            "uid":       uid,
            "trainId":   train_id,
            "ssd":       ssd_date,
            "toc":       toc,
            "stop_type": child.get("tag"),
            "tpl":       stop.get("tpl"),
            "platform":  stop.get("plat"),
            "act":       stop.get("act"),
            "pta":       stop.get("pta"),
            "ptd":       stop.get("ptd"),
            "wta":       stop.get("wta"),
            "wtd":       stop.get("wtd"),
            "wtp":       stop.get("wtp"),
        })
    return rows