"""File parsing utilities - CSV/JSON loader and Darwin timetable parser."""

import json

import pandas as pd


def load_file(path):
    """Load a CSV or newline-delimited JSON file into a DataFrame."""
    if path.endswith(".csv"):
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty CSV file: {path}")
            return pd.DataFrame()   # return empty df
    else:
        return pd.read_json(path, lines=True)



def parse_darwin_timetable_files(timetable_files):
    """Read one or more JSON timetable files and return a flat DataFrame."""
    rows = []
    for file_path in timetable_files:
        with open(file_path, "r") as f:
            data = json.load(f)

        journeys = data if isinstance(data, list) else [data]
        print(f"Processing {file_path} with {len(journeys)} journeys")

        for journey in journeys:
            rows.extend(_journey_to_rows(journey))

    return pd.DataFrame(rows)


def _journey_to_rows(journey):
    """Flatten a single journey dict into one row per stop."""
    attrs = journey.get("attributes", {})
    rid = attrs.get("rid")
    uid = attrs.get("uid")
    train_id = attrs.get("trainId")
    ssd = attrs.get("ssd")
    toc = attrs.get("toc")

    rows = []
    for child in journey.get("children", []):
        stop = child.get("attributes", {})
        rows.append({
            "rid": rid,
            "uid": uid,
            "trainId": train_id,
            "ssd": pd.to_datetime(ssd).date(),
            "toc": toc,
            "stop_type": child.get("tag"),
            "tpl": stop.get("tpl"),
            "platform": stop.get("plat"),
            "act": stop.get("act"),
            "pta": stop.get("pta"),
            "ptd": stop.get("ptd"),
            "wta": stop.get("wta"),
            "wtd": stop.get("wtd"),
            "wtp": stop.get("wtp"),
        })
    return rows



def parse_service_details(data):
    details = data["serviceAttributesDetails"]

    rid = details["rid"]
    toc = details["toc_code"]
    date = details["date_of_service"]

    for loc in details["locations"]:
        record = {
            "rid": rid,
            "date": date,
            "toc": toc,
            "location": loc["location"],
            "planned_dep": loc.get("gbtt_ptd"),
            "actual_dep": loc.get("actual_td"),
            "planned_arr": loc.get("gbtt_pta"),
            "actual_arr": loc.get("actual_ta"),
            "cancel_reason": loc.get("late_canc_reason")
        }
    return pd.DataFrame([record])

