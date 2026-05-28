"""Feature engineering - timetable reshaping used by EDA 06."""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Timetable schedule reshaping
# ---------------------------------------------------------------------------

def reshape_timetable_to_schedule(timetable_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Map each timetable stop to a station and derive a planned_timestamp.

    Resolves the working time (wta/wtd/wtp) to use based on the activity
    flag (act), combines it with the service date (ssd) to produce a
    UTC-naive planned_timestamp, and maps the TIPLOC code to station
    identifiers via stations_df.

    Returns a DataFrame with columns:
        ssd, tpl, station_code, stanox, station_name,
        timetable_train_id, planned_timestamp, event_type
    """
    df = timetable_df.copy()

    def _clean_act(x) -> str:
        """Normalise the act field to a clean string; return '' for null/junk."""
        if isinstance(x, str):
            x = x.strip()
            return "" if x.lower() in ("nan", "none", "null", "") else x
        return ""

    # Build TIPLOC → station identifier lookup
    tpl_lookup = (
        stations_df[["tiploc", "tlc", "stanox", "station_name"]]
        .drop_duplicates(subset="tiploc")
        .rename(columns={"tlc": "station_code"})
        .set_index("tiploc")
    )

    df["station_code"]      = df["tpl"].map(tpl_lookup["station_code"])
    df["stanox"]            = df["tpl"].map(tpl_lookup["stanox"])
    df["station_name"]      = df["tpl"].map(tpl_lookup["station_name"])
    df["timetable_train_id"] = df["trainId"]
    df["act"]               = df["act"].apply(_clean_act)
    df['rid']               = df['rid'].astype(str).str.strip()
    df['stop_type']         = df['stop_type'].astype(str).str.strip()
    keep_cols = ["ssd", "tpl", "station_code", "stanox", "station_name", "timetable_train_id",'rid','stop_type']

    def _choose_time(row) -> str:
        """Select the working time field based on the activity flag."""
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
            print(f"choose_time error: {row} - {e}")
            primary = None
        return primary or row["wta"] or row["wtd"] or row["wtp"]

    df["chosen_time"] = df.apply(_choose_time, axis=1)

    # Pad HH:MM → HH:MM:SS
    df["chosen_time"] = df["chosen_time"].astype(str).str.strip()
    df["chosen_time"] = np.where(
        df["chosen_time"].str.len() == 5,
        df["chosen_time"] + ":00",
        df["chosen_time"],
    )

    df["planned_timestamp"] = pd.to_datetime(
        df["ssd"].astype(str) + " " + df["chosen_time"], errors="coerce"
    )

    df["event_type"] = np.select(
        [
            df["act"].str.contains("TF", na=False),
            df["act"].str.contains("TB", na=False),
            (df["wtp"].notna() & ~df["act"].str.contains(r"T|D|U", na=False)),
            df["act"].str.contains(r"T|D|U", na=False),
        ],
        ["ARRIVAL", "DEPARTURE", "PASS", "ARRIVAL"],
        default="DEPARTURE",
    )

    return df[keep_cols + ["planned_timestamp", "event_type"]]