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

import time

from multiprocessing import Pool, cpu_count
import pandas as pd

def parse_darwin_timetable_files_parallel(timetable_files):
    
    with Pool(cpu_count()) as p:
        dfs = p.map(parse_one_file, timetable_files)

    
    return pd.concat(dfs, ignore_index=True)

def parse_one_file(file_path):
    
    with open(file_path, "r") as f:
        data = json.load(f)

    journeys = data if isinstance(data, list) else [data]
    
    print(f"Processing {file_path} with {len(journeys)} journeys")
    rows = []
    for journey in journeys:
        rows.extend(_journey_to_rows(journey))
    return pd.DataFrame(rows)

def parse_darwin_timetable_files(timetable_files):
    """Read one or more JSON timetable files and return a flat DataFrame."""
    rows = []

    start_total = time.time()
    
    for file_path in timetable_files:
        
        if file_path == "data\darwin_timetable\PPTimetable_20260427020459_v8.json":
            t0 = time.time()
    
            with open(file_path, "r") as f:
                data = json.load(f)
            print("Load:", time.time() - t0)

            t1 = time.time()
        
            journeys = data if isinstance(data, list) else [data]
            print(f"Processing {file_path} with {len(journeys)} journeys")

            print("Print length:", time.time() - t1)

            t2 = time.time()
            
            for journey in journeys:
                rows.extend(_journey_to_rows(journey))
        
            print("Parse:", time.time() - t2)
        else:
            continue

    print("Total:", time.time() - start_total)   
    return pd.DataFrame(rows)


def _journey_to_rows(journey):
    """Flatten a single journey dict into one row per stop."""
    attrs = journey.get("attributes", {})
    rid = attrs.get("rid")
    uid = attrs.get("uid")
    train_id = attrs.get("trainId")
    ssd = attrs.get("ssd")
    ssd_date = pd.to_datetime(ssd).date()
    toc = attrs.get("toc")

    rows = []
    for child in journey.get("children", []):
        stop = child.get("attributes", {})
        rows.append({
            "rid": rid,
            "uid": uid,
            "trainId": train_id,
            "ssd": ssd_date,
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

