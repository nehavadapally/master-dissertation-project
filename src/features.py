"""Feature engineering — timetable reshaping, temporal filtering, delay calculation."""

import pandas as pd


# ---------------------------------------------------------------------------
# Timetable schedule reshaping
# ---------------------------------------------------------------------------

EVENT_MAP = {"wta": "ARRIVAL", "wtd": "DEPARTURE", "wtp": "PASS"}
TIME_COLS = ["wta", "wtd", "wtp"]


def reshape_timetable_to_schedule(timetable_df, stations_lookup_df):
    """Convert wide timetable (wta/wtd/wtp) into long schedule rows.

    Adds station_code and stanox via TIPLOC lookup, melts time columns
    into individual event rows, and builds full planned_timestamp.

    Returns:
        DataFrame with columns: stanox, tpl, station_code, timetable_train_id,
        planned_timestamp, event_type.
    """
    tpl_lookup = (
        stations_lookup_df[["tiploc", "3alpha", "stanox"]]
        .drop_duplicates(subset="tiploc")
        .rename(columns={"3alpha": "station_code"})
    )
    tpl_to_code = tpl_lookup.set_index("tiploc")["station_code"]
    tpl_to_stanox = tpl_lookup.set_index("tiploc")["stanox"]

    df = timetable_df.copy()
    df["station_code"] = df["tpl"].map(tpl_to_code)
    df["stanox"] = df["tpl"].map(tpl_to_stanox)
    df["timetable_train_id"] = df["trainId"]

    keep_cols = ["ssd", "tpl", "station_code", "stanox", "timetable_train_id"]

    # Melt wta/wtd/wtp into long format
    df = df.melt(
        id_vars=keep_cols,
        value_vars=[c for c in TIME_COLS if c in df.columns],
        var_name="event_col",
        value_name="time_str",
    )
    df = df.dropna(subset=["time_str"])

    # Normalise HH:MM → HH:MM:00
    df["time_str"] = df["time_str"].apply(
        lambda x: f"{x}:00" if len(str(x)) == 5 else str(x)
    )

    # Build full datetime
    df["planned_timestamp"] = pd.to_datetime(
        df["ssd"].astype(str) + " " + df["time_str"], errors="coerce"
    )
    df["event_type"] = df["event_col"].map(EVENT_MAP)

    return df[keep_cols + ["planned_timestamp", "event_type"]]


# ---------------------------------------------------------------------------
# Merge + filter
# ---------------------------------------------------------------------------

def merge_on_station(left_df, right_df, join_col="station_code"):
    """Inner-join two DataFrames on a station column."""
    return left_df.merge(right_df, on=join_col, how="inner")


def merge_schedule_with_closures(expanded_road_df, schedule_df):
    """Merge expanded road closures with schedule on stanox, resolving suffix conflicts."""
    merged = expanded_road_df.merge(
        schedule_df, on="stanox", how="inner",
        suffixes=("_road", "_schedule"),
    )

    # Keep schedule versions of duplicated columns
    merged["tpl"] = merged["tpl_schedule"]
    merged["station_code"] = merged["station_code_schedule"]
    merged = merged.drop(
        columns=["tpl_road", "tpl_schedule", "station_code_road", "station_code_schedule"]
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
