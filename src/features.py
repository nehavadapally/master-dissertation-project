"""Feature engineering - timetable reshaping, temporal filtering, delay calculation."""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Timetable schedule reshaping
# ---------------------------------------------------------------------------

EVENT_MAP = {"wta": "ARRIVAL", "wtd": "DEPARTURE", "wtp": "PASS"}

def reshape_timetable_to_schedule(timetable_df, stations_df):

    df = timetable_df.copy()
    def clean_act(x):
        if isinstance(x, str):
            x = x.strip()
            if x.lower() in ("nan", "none", "null", ""):
                return ""
            return x
        return ""   # for floats, NaN, None, objects

    # Lookup maps
    tpl_lookup = (
        stations_df[["tiploc", "tlc", "stanox","station"]]
        .drop_duplicates(subset="tiploc")
        .rename(columns={"tlc": "station_code","station": "station_name"})
    )

    df["station_code"] = df["tpl"].map(tpl_lookup.set_index("tiploc")["station_code"])
    df["stanox"] = df["tpl"].map(tpl_lookup.set_index("tiploc")["stanox"])
    df["station_name"] = df["tpl"].map(tpl_lookup.set_index("tiploc")["station_name"])
    df["timetable_train_id"] = df["trainId"]
    keep_cols = ["ssd", "tpl", "station_code", "stanox", "station_name", "timetable_train_id"]
    df["act"] = df["act"].apply(clean_act)

    # print(df["act"].dtype)
    def choose_time(row):
        act = row["act"]

        try:
            match True:
            
                case _ if "TF" in act:
                    primary = row["wta"]
                case _ if "TB" in act:
                    primary = row["wtd"]
                case _ if ("T" not in act and "D" not in act and "U" not in act and row["wtp"]):
                    primary = row["wtp"]
                case _ if ("T" in act or "D" in act or "U" in act):
                    primary = row["wta"]
                case _:
                    primary = row["wtd"]
                
        except Exception as e:
            print(row, e)

        # fallback chain
        return primary or row["wta"] or row["wtd"] or row["wtp"]

    df["chosen_time"] = df.apply(choose_time, axis=1)

    # Normalise
    df["chosen_time"] = df["chosen_time"].astype(str).str.strip()
    df["chosen_time"] = np.where(
        df["chosen_time"].str.len() == 5,
        df["chosen_time"] + ":00",
        df["chosen_time"]
    )

    df["planned_timestamp"] = pd.to_datetime(
        df["ssd"].astype(str) + " " + df["chosen_time"],
        errors="coerce"
    )

    # Event type
    df["event_type"] = np.select(
        [
            df["act"].str.contains("TF", na=False),
            df["act"].str.contains("TB", na=False),
            (df["wtp"].notna() & ~df["act"].str.contains(r"T|D|U", na=False)),
            df["act"].str.contains(r"T|D|U", na=False)
        ],
        ["ARRIVAL", "DEPARTURE", "PASS", "ARRIVAL"],
        default="DEPARTURE"
    )

    return df[keep_cols + ["planned_timestamp", "event_type"]]


# ---------------------------------------------------------------------------
# Merge + filter
# ---------------------------------------------------------------------------

def merge_on_station(left_df, right_df, join_col="station_code"):
    """Inner-join two DataFrames on a station column."""
    df = left_df.merge(right_df, left_on=["start_date",join_col], right_on = ['actual_date',join_col], how="inner")
    df = df.rename(columns={"station_name_x": "station_name"}).drop(columns=["station_name_y"])
    return df


def merge_schedule_with_closures(expanded_road_df, schedule_df):
    """Merge expanded road closures with schedule on stanox, resolving suffix conflicts."""
    merged = expanded_road_df.merge(
        schedule_df, left_on=['start_date',"stanox"], right_on = ['ssd','stanox'], how="inner",
        suffixes=("_road", "_schedule"),
    )
    # rename schedule versions of duplicated columns
    merged = merged.rename(columns= {"station_name_schedule": "station_name", "station_code_schedule": "station_code", "tpl_schedule": "tpl"} )
    merged = merged.drop(
        columns=["tpl_road", "station_code_road", "station_name_road",]
    )
    return merged


def filter_within_time_window(df, window_minutes=60):
    """Keep rows where planned_timestamp is within (0, window] minutes after closure_start_time.

    Adds a planned_time_diff column (minutes).
    """
    df = df.copy()
    df["closure_start_time"] = pd.to_datetime(df["closure_start_time"])
    df["planned_timestamp"] = pd.to_datetime(df["planned_timestamp"])

    df["planned_time_diff"] = (
        (df["planned_timestamp"] - df["closure_start_time"]).dt.total_seconds() / 60
    )

    mask = (df["planned_time_diff"] > 0) & (df["planned_time_diff"] <= window_minutes)
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Delay calculation
# ---------------------------------------------------------------------------

def add_delay_column(df, actual_col="actual_timestamp", planned_col="planned_timestamp"):
    """Add delay (minutes) = actual - planned."""
    df = df.copy()
    df[actual_col] = pd.to_datetime(df[actual_col])
    df[planned_col] = pd.to_datetime(df[planned_col])
    df["delay"] = (df[actual_col] - df[planned_col]).dt.total_seconds() / 60
    return df
