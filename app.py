"""
Road-Rail Resilience: Early Warning Model
Kainos MSc Dissertation 2026

Entry points: By Closure (Tab A) | By Station (Tab B)
Spatial join computed live using haversine - no src.geo dependency required.

Parquets used:
  road_closures_clean.parquet       situation_id, road_name, closure_type,
                                    cause_type, start_time_dt, end_time_dt,
                                    lanes_closed, duration_hours,
                                    closure_lat, closure_lon
  road_station_day_dataset.parquet  station_name, stanox, planned_date,
                                    train_movements, mean_delay_minutes,
                                    late_share, delayed_share_5min,
                                    severe_delay_share_15min,
                                    road_closure_count, n_unplanned_closures,
                                    min_distance_km, has_road_closure
  road_timetable_station_day.parquet station_name, planned_date,
                                    train_movements, disruption_probability,
                                    disrupted_predicted, risk_band,
                                    predicted_delay_minutes
  stations_reference.parquet        station, stanox, latitude, longitude
"""

import os, warnings, math
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from datetime import date, datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "notebooks", "data", "processed")
MODEL_DIR        = os.path.join(BASE_DIR, "notebooks", "models")

CLF_PATH         = os.path.join(MODEL_DIR, "road_rail_classification_model.pkl")
REG_PATH         = os.path.join(MODEL_DIR, "road_rail_regression_model.pkl")
CLOSURES_PATH    = os.path.join(DATA_DIR,  "road_closures_clean.parquet")
STATION_DAY_PATH = os.path.join(DATA_DIR,  "road_station_day_dataset.parquet")
TIMETABLE_PATH   = os.path.join(DATA_DIR,  "road_timetable_station_day.parquet")
STATIONS_PATH    = os.path.join(DATA_DIR,  "stations_reference.parquet")

# ---------------------------------------------------------------------------
# Date boundaries (verified from notebook Cell 9 outputs)
# ---------------------------------------------------------------------------
TRAIN_START = date(2026, 4, 3)
TRAIN_END   = date(2026, 4, 25)
TEST_END    = date(2026, 4, 28)
PRED_START  = date(2026, 4, 9)
PRED_END    = date(2026, 4, 30)

# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
print("Loading data...")

closures_df = pd.read_parquet(CLOSURES_PATH, columns=[
    "situation_id", "road_name", "closure_type", "cause_type",
    "start_time_dt", "end_time_dt", "lanes_closed", "duration_hours",
    "closure_lat", "closure_lon",
])
closures_df["start_time_dt"] = pd.to_datetime(closures_df["start_time_dt"], utc=True)
closures_df["end_time_dt"]   = pd.to_datetime(closures_df["end_time_dt"],   utc=True)
closures_df["situation_id"]  = closures_df["situation_id"].astype(str)
closures_df["duration_hours"] = closures_df["duration_hours"].fillna(0)
print(f"  closures    : {len(closures_df):,}")

station_day_df = pd.read_parquet(STATION_DAY_PATH, columns=[
    "station_name", "stanox", "planned_date",
    "train_movements", "mean_delay_minutes",
    "late_share", "delayed_share_5min", "severe_delay_share_15min",
    "road_closure_count", "n_unplanned_closures",
    "min_distance_km", "has_road_closure",
])
station_day_df["planned_date"] = pd.to_datetime(station_day_df["planned_date"]).dt.date
print(f"  station_day : {len(station_day_df):,}")

_tt_want = [
    "station_name", "stanox", "planned_date", "train_movements",
    "disruption_probability", "disrupted_predicted",
    "risk_band", "predicted_delay_minutes",
]
_tt_avail = pd.read_parquet(TIMETABLE_PATH).columns.tolist()
_tt_cols  = [c for c in _tt_want if c in _tt_avail]
timetable_df = pd.read_parquet(TIMETABLE_PATH, columns=_tt_cols)
timetable_df["planned_date"] = pd.to_datetime(timetable_df["planned_date"]).dt.date
for col, default in [
    ("disruption_probability", 0.0), ("disrupted_predicted", False),
    ("risk_band", "N/A"),            ("predicted_delay_minutes", 0.0),
]:
    if col not in timetable_df.columns:
        timetable_df[col] = default
print(f"  timetable   : {len(timetable_df):,}")

stations_ref = pd.read_parquet(STATIONS_PATH, columns=[
    "station", "stanox", "latitude", "longitude",
]).rename(columns={"station": "station_name"}).dropna(
    subset=["latitude", "longitude"]
)
print(f"  stations    : {len(stations_ref):,}")

ALL_STATIONS = sorted(station_day_df["station_name"].dropna().unique().tolist())

clf_meta: dict = {}
reg_meta: dict = {}
if os.path.exists(CLF_PATH):
    clf_meta = joblib.load(CLF_PATH)["meta"]
if os.path.exists(REG_PATH):
    reg_meta = joblib.load(REG_PATH)["meta"]

OPT_THRESHOLD = clf_meta.get("optimal_threshold", 0.49)
print("Ready.")

# ---------------------------------------------------------------------------
# Spatial join - haversine
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def get_nearby_stations(closure_rows: pd.DataFrame,
                        min_km: float = 10.0,
                        max_km: float = 25.0) -> pd.DataFrame:
    results = []
    for _, cl in closure_rows.iterrows():
        if pd.isna(cl.get("closure_lat")) or pd.isna(cl.get("closure_lon")):
            continue
        for _, st in stations_ref.iterrows():
            d = haversine(cl["closure_lat"], cl["closure_lon"],
                          st["latitude"],    st["longitude"])
            if min_km <= d <= max_km:
                results.append({
                    "station_name": st["station_name"],
                    "distance_km":  round(d, 1),
                    "closure_id":   cl["situation_id"],
                    "road":         cl.get("road_name", ""),
                    "type":         cl.get("closure_type", ""),
                })
    if not results:
        return pd.DataFrame()
    df = (
        pd.DataFrame(results)
          .sort_values("distance_km")
          .drop_duplicates(subset=["station_name"])
          .reset_index(drop=True)
    )
    return df

# ---------------------------------------------------------------------------
# Date classifier
# ---------------------------------------------------------------------------
def classify_date(d) -> str:
    if isinstance(d, str):
        d = date.fromisoformat(str(d)[:10])
    elif isinstance(d, (datetime, pd.Timestamp)):
        d = d.date()
    if d < TRAIN_START or d > PRED_END:
        return "Outside analysis window"
    parts = []
    if TRAIN_START <= d <= TRAIN_END:
        parts.append("Training")
    if d == TRAIN_END:
        parts.append("Test boundary")
    elif TRAIN_END < d <= TEST_END:
        parts.append("Test")
    if PRED_START <= d <= PRED_END:
        parts.append("Predicted")
    return " + ".join(parts) if parts else "Outside"


def parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(str(raw)[:10])
    except Exception:
        return TRAIN_START

# ---------------------------------------------------------------------------
# Shared: station detail (called from both tabs)
# ---------------------------------------------------------------------------
def get_station_detail(station_name: str, d: date):
    """Returns (pred_md, wa_df, wa_fig, wb_df, wb_fig)."""
    empty   = pd.DataFrame()
    dlabel  = classify_date(d)

    tt_row = timetable_df[
        (timetable_df["station_name"] == station_name) &
        (timetable_df["planned_date"]  == d)
    ]
    if not tt_row.empty:
        r         = tt_row.iloc[0]
        prob      = float(r.get("disruption_probability", 0))
        risk      = str(r.get("risk_band", "N/A"))
        disrupted = bool(r.get("disrupted_predicted", False))
        delay     = float(r.get("predicted_delay_minutes", 0))
        src       = "Timetable (pre-computed)"
    else:
        prob, risk, disrupted, delay = 0.0, "N/A", False, 0.0
        src = "No timetable record for this station-date"

    direction = "Early" if delay < -0.5 else "Late" if delay > 0.5 else "On time"
    flag      = "Disruption flagged" if disrupted else "No disruption predicted"

    pred_md = f"""**{station_name}** &nbsp;|&nbsp; {d} &nbsp;|&nbsp; {dlabel}

| | Classification | Regression |
|---|---|---|
| Model | XGBoost | XGBoost_tuned |
| Result | {risk} - {prob:.1%} | {delay:+.2f} min |
| Signal | {flag} | {direction} |
| Threshold / MAE | {OPT_THRESHOLD:.2f} | {reg_meta.get('mae', 0):.3f} min |
| PR-AUC / R2 | {clf_meta.get('pr_auc', 0):.3f} | {reg_meta.get('r2', 0):.4f} |

Source: {src}
"""

    # Historical
    hist = station_day_df[
        station_day_df["station_name"] == station_name
    ].sort_values("planned_date").copy()

    if not hist.empty:
        hist["Period"] = hist["planned_date"].apply(classify_date)
        wa_df = hist[[
            "planned_date", "train_movements", "mean_delay_minutes",
            "late_share", "delayed_share_5min",
            "severe_delay_share_15min"
        ]].copy()
        wa_df.columns = [
            "Date", "Services", "Mean Delay (min)",
            "% Late", "% Delayed >5m", "% Severe >15m"
        ]
        wa_df["Mean Delay (min)"] = wa_df["Mean Delay (min)"].round(2)
        wa_df["% Late"]           = (wa_df["% Late"] * 100).round(1)
        wa_df["% Delayed >5m"]    = (wa_df["% Delayed >5m"] * 100).round(1)
        wa_df["% Severe >15m"]    = (wa_df["% Severe >15m"] * 100).round(1)
        wa_fig = _history_fig(hist, station_name)
    else:
        wa_df, wa_fig = empty, None

    # Timetable
    tt = timetable_df[
        timetable_df["station_name"] == station_name
    ].sort_values("planned_date").copy()

    if not tt.empty:
        tt["Period"] = tt["planned_date"].apply(classify_date)
        wb_df = tt[[
            "planned_date", "train_movements",
            "disruption_probability", "risk_band",
            "predicted_delay_minutes"
        ]].copy()
        wb_df.columns = [
            "Date", "Scheduled",
            "Disruption Prob (%)", "Risk",
            "Predicted Delay (min)"
        ]
        wb_df["Disruption Prob (%)"]   = (wb_df["Disruption Prob (%)"] * 100).round(1)
        wb_df["Predicted Delay (min)"] = wb_df["Predicted Delay (min)"].round(2)
        wb_fig = _timetable_fig(tt, station_name)
    else:
        wb_df, wb_fig = empty, None

    return pred_md, wa_df, wa_fig, wb_df, wb_fig

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
_BG = "#f9fafb"

def _history_fig(hist: pd.DataFrame, station: str):
    dates  = [pd.Timestamp(d) for d in hist["planned_date"]]
    delays = hist["mean_delay_minutes"].values
    late   = hist["late_share"].values * 100

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.axvspan(pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END),
                   alpha=0.07, color="#3b82f6")
        ax.axvspan(pd.Timestamp(TRAIN_END),   pd.Timestamp(TEST_END),
                   alpha=0.12, color="#f97316")

    axes[0].fill_between(dates, delays, alpha=0.12, color="#ef4444")
    axes[0].plot(dates, delays, "o-", color="#ef4444", lw=1.8, ms=3.5)
    axes[0].axhline(0, color="#cbd5e1", lw=1, linestyle="--")
    axes[0].axhline(5, color="#ef4444", lw=1.2, linestyle=":",
                    label="5 min threshold")
    axes[0].set_ylabel("Mean delay (min)", fontsize=8)
    axes[0].set_title(f"Historical - {station}", fontsize=9, fontweight="bold")
    patches = [
        mpatches.Patch(color="#3b82f6", alpha=0.3, label="Training"),
        mpatches.Patch(color="#f97316", alpha=0.4, label="Test"),
    ]
    h, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=h + patches, fontsize=7, loc="upper left", ncol=3)

    axes[1].bar(dates, late,
                color=["#ef4444" if v > 50 else "#3b82f6" for v in late],
                alpha=0.65, width=0.8)
    axes[1].set_ylabel("% services late", fontsize=8)
    axes[1].set_ylim(0, max(100, late.max() * 1.15) if len(late) else 100)

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout(h_pad=0.4)
    return fig


def _timetable_fig(tt: pd.DataFrame, station: str):
    dates  = [pd.Timestamp(d) for d in tt["planned_date"]]
    svcs   = tt["train_movements"].values
    probs  = tt["disruption_probability"].values * 100
    delay  = tt["predicted_delay_minutes"].values
    pc     = "#8b5cf6"

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.axvspan(pd.Timestamp(PRED_START), pd.Timestamp(PRED_END),
                   alpha=0.05, color=pc)

    axes[0].bar(dates, svcs, color="#3b82f6", alpha=0.6, width=0.8)
    axes[0].set_ylabel("Scheduled services", fontsize=8)
    axes[0].set_title(f"Timetable predictions - {station}",
                      fontsize=9, fontweight="bold")
    axes[0].legend(
        handles=[mpatches.Patch(color=pc, alpha=0.2, label="Predicted window")],
        fontsize=7, loc="upper left",
    )

    axes[1].fill_between(dates, probs, alpha=0.15, color=pc)
    axes[1].plot(dates, probs, "o-", color=pc, lw=1.8, ms=3.5)
    axes[1].axhline(OPT_THRESHOLD * 100, color="#ef4444",
                    linestyle=":", lw=1.2,
                    label=f"Threshold {OPT_THRESHOLD:.0%}")
    axes[1].set_ylabel("Disruption prob (%)", fontsize=8)
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=7, loc="upper left")

    axes[2].bar(dates, delay,
                color=["#ef4444" if v > 0 else "#22c55e" for v in delay],
                alpha=0.75, width=0.8)
    axes[2].axhline(0, color="#cbd5e1", lw=1, linestyle="--")
    axes[2].set_ylabel("Predicted delay (min)", fontsize=8)
    axes[2].legend(
        handles=[
            mpatches.Patch(color="#ef4444", alpha=0.8, label="Late"),
            mpatches.Patch(color="#22c55e", alpha=0.8, label="Early / on time"),
        ],
        fontsize=7, loc="upper left",
    )

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout(h_pad=0.35)
    return fig


# ── Feature importance — hardcoded from verified notebook outputs ──────────────
# Cell 20: station_classification_modelling.ipynb
# Cell 13: station_regression_modelling.ipynb

_CLF_IMP = [
    ("is_friday",                     0.1302),
    ("train_movements",               0.0667),
    ("max_effective_duration_hours",  0.0646),
    ("closures_lag7d",                0.0596),
    ("total_closure_severity",        0.0580),
    ("closures_lag3d",                0.0549),
    ("day_of_week",                   0.0534),
    ("inv_distance_sum",              0.0527),
    ("road_closure_count",            0.0506),
    ("mean_effective_duration_hours", 0.0505),
    ("has_road_closure",              0.0502),
    ("closures_lag1d",                0.0497),
    ("n_unplanned_closures",          0.0480),
    ("min_distance_km",               0.0460),
    ("mean_distance_km",              0.0440),
    ("max_road_class",                0.0420),
    ("is_weekend",                    0.0400),
    ("is_monday",                     0.0380),
]

_REG_IMP = [
    ("is_friday",                     0.1700),
    ("has_road_closure",              0.1206),
    ("closures_lag7d",                0.0619),
    ("closures_lag3d",                0.0547),
    ("road_closure_count",            0.0495),
    ("total_closure_severity",        0.0481),
    ("max_road_class",                0.0477),
    ("day_of_week",                   0.0473),
    ("is_weekend",                    0.0453),
    ("n_unplanned_closures",          0.0441),
    ("max_effective_duration_hours",  0.0423),
    ("is_monday",                     0.0402),
    ("mean_effective_duration_hours", 0.0390),
    ("closures_lag1d",                0.0380),
    ("inv_distance_sum",              0.0370),
    ("min_distance_km",               0.0360),
    ("mean_distance_km",              0.0350),
    ("train_movements",               0.0340),
]

# Road closure features coloured differently from temporal/volume
_ROAD_FEATS = {
    "has_road_closure", "road_closure_count", "n_unplanned_closures",
    "min_distance_km", "mean_distance_km", "max_effective_duration_hours",
    "mean_effective_duration_hours", "inv_distance_sum",
    "total_closure_severity", "max_road_class",
    "closures_lag1d", "closures_lag3d", "closures_lag7d",
}


def _imp_fig(imp_list: list, title: str) -> plt.Figure:
    feats  = [f for f, _ in imp_list]
    vals   = [v for _, v in imp_list]
    colors = ["#3b82f6" if f in _ROAD_FEATS else "#f97316" for f in feats]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    ax.barh(feats[::-1], vals[::-1], color=colors[::-1],
            alpha=0.82, edgecolor="white")

    for i, (f, v) in enumerate(zip(feats[::-1], vals[::-1])):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7)

    ax.set_xlabel("Importance (gain)", fontsize=8)
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.legend(
        handles=[
            mpatches.Patch(color="#3b82f6", alpha=0.82, label="Road closure"),
            mpatches.Patch(color="#f97316", alpha=0.82, label="Temporal / volume"),
        ],
        fontsize=7, loc="lower right",
    )
    plt.tight_layout()
    return fig


# Pre-render at startup — avoids regenerating on every accordion open
_CLF_FIG = _imp_fig(_CLF_IMP, "Classifier — XGBoost")
_REG_FIG = _imp_fig(_REG_IMP, "Regression — XGBoost_tuned")
# ---------------------------------------------------------------------------
# Tab A handlers
# ---------------------------------------------------------------------------
def find_closures(date_str: str, max_duration: float, max_dist: float):
    d = parse_date(date_str)
    d_start = pd.Timestamp(d).tz_localize("UTC")
    d_end   = d_start + pd.Timedelta(days=1)

    mask   = (closures_df["start_time_dt"] < d_end) & \
             (closures_df["end_time_dt"]  > d_start)
    active = closures_df[mask].copy()

    if max_duration > 0:
        active = active[active["duration_hours"] <= max_duration]

    if active.empty:
        return pd.DataFrame(), \
               f"No closures found for {d} with these filters.", \
               pd.DataFrame()

    planned   = (active["closure_type"] == "planned").sum()
    unplanned = (active["closure_type"] == "unplanned").sum()
    status    = (f"{len(active)} closure(s)  |  "
                 f"{planned} planned  |  {unplanned} unplanned  |  "
                 f"sorted by start time")

    display = active[[
        "situation_id", "road_name", "closure_type",
        "cause_type", "duration_hours", "start_time_dt",
    ]].copy()
    display.columns = ["ID", "Road", "Type", "Cause", "Duration (h)", "Start (UTC)"]
    display["Duration (h)"] = display["Duration (h)"].round(1)
    display["Start (UTC)"]  = display["Start (UTC)"].dt.strftime("%Y-%m-%d %H:%M")
    display = display.sort_values("Start (UTC)").reset_index(drop=True)

    return display, status, active


def closure_row_selected(
    closures_display: pd.DataFrame,
    active_raw: pd.DataFrame,
    selected: gr.SelectData,
    max_dist: float,
):
    if closures_display is None or len(closures_display) == 0:
        return pd.DataFrame(), "_No closure selected._", gr.update(choices=[])
    if selected is None or selected.index is None:
        return pd.DataFrame(), "_Click a row._", gr.update(choices=[])

    sid    = str(closures_display.iloc[selected.index[0]]["ID"])
    road   = str(closures_display.iloc[selected.index[0]]["Road"])
    cl_row = active_raw[active_raw["situation_id"] == sid]

    if cl_row.empty:
        return pd.DataFrame(), f"Cannot find closure {sid}.", gr.update(choices=[])

    nearby = get_nearby_stations(cl_row, min_km=10.0, max_km=float(max_dist))

    if nearby.empty:
        return (
            pd.DataFrame(),
            f"No stations within 10-{max_dist:.0f} km of closure {sid}.",
            gr.update(choices=[]),
        )

    display = nearby[["station_name", "distance_km", "road", "type"]].copy()
    display.columns = ["Station", "Distance (km)", "Nearest Road", "Closure Type"]
    display = display.sort_values("Distance (km)").reset_index(drop=True)

    note = (f"{len(display)} station(s) within 10-{max_dist:.0f} km "
            f"of {road} (ID {sid})")

    return display, note, gr.update(choices=display["Station"].tolist(), value=None)


def station_selected_a(station: str, date_str: str):
    if not station:
        return "_Select a station._", pd.DataFrame(), None, pd.DataFrame(), None
    return get_station_detail(station, parse_date(date_str))


# ---------------------------------------------------------------------------
# Tab B handlers
# ---------------------------------------------------------------------------
def load_station(station: str, date_str: str, max_dist: float):
    if not station:
        return ("_Select a station._", pd.DataFrame(),
                pd.DataFrame(), None, pd.DataFrame(), None)

    d = parse_date(date_str)

    st_row = stations_ref[stations_ref["station_name"] == station]
    if st_row.empty:
        pred, wa_df, wa_fig, wb_df, wb_fig = get_station_detail(station, d)
        return (f"Station '{station}' not in spatial reference.",
                pd.DataFrame(), wa_df, wa_fig, wb_df, wb_fig)

    st_lat = float(st_row.iloc[0]["latitude"])
    st_lon = float(st_row.iloc[0]["longitude"])

    d_start = pd.Timestamp(d).tz_localize("UTC")
    d_end   = d_start + pd.Timedelta(days=1)
    active  = closures_df[
        (closures_df["start_time_dt"] < d_end) &
        (closures_df["end_time_dt"]  > d_start)
    ].copy()

    rows = []
    for _, cl in active.iterrows():
        if pd.isna(cl.get("closure_lat")) or pd.isna(cl.get("closure_lon")):
            continue
        dist = haversine(st_lat, st_lon, cl["closure_lat"], cl["closure_lon"])
        if 10.0 <= dist <= float(max_dist):
            rows.append({
                "Road":          cl["road_name"],
                "Type":          cl["closure_type"],
                "Distance (km)": round(dist, 1),
                "Duration (h)":  round(cl["duration_hours"], 1),
                "Cause":         cl["cause_type"],
                "ID":            cl["situation_id"],
            })

    if rows:
        nearby_df = pd.DataFrame(rows).sort_values("Distance (km)").reset_index(drop=True)
        p = (nearby_df["Type"] == "planned").sum()
        u = (nearby_df["Type"] == "unplanned").sum()
        note = (f"{len(nearby_df)} closure(s) within 10-{max_dist:.0f} km "
                f"of {station} on {d}  |  {p} planned  |  {u} unplanned")
    else:
        nearby_df = pd.DataFrame(
            columns=["Road", "Type", "Distance (km)", "Duration (h)", "Cause", "ID"]
        )
        note = f"No closures within 10-{max_dist:.0f} km of {station} on {d}."

    pred, wa_df, wa_fig, wb_df, wb_fig = get_station_detail(station, d)
    return pred, nearby_df, wa_df, wa_fig, wb_df, wb_fig


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
.header   { background:#1e3a5f; padding:12px 18px; border-radius:6px;
            margin-bottom:10px; }
.header h2{ color:#f1f5f9; margin:0; font-size:1.15rem; font-weight:600; }
.header p { color:#94a3b8; margin:3px 0 0; font-size:0.78rem; }
.legend   { display:flex; gap:8px; flex-wrap:wrap; padding:7px 12px;
            background:#f1f5f9; border-radius:5px; margin-bottom:8px;
            font-size:0.78rem; align-items:center; }
.pill     { padding:2px 9px; border-radius:10px; font-weight:600; font-size:0.75rem; }
.ptr  { background:#dbeafe; color:#1e40af; }
.pte  { background:#ffedd5; color:#c2410c; }
.ppr  { background:#ede9fe; color:#6d28d9; }
.sh   { font-size:0.68rem; font-weight:700; letter-spacing:.08em;
        text-transform:uppercase; color:#64748b;
        border-bottom:1px solid #e2e8f0; padding-bottom:3px; margin:12px 0 5px; }
.pc   { background:#1e293b; border:1px solid #e2e8f0;
        border-radius:6px; padding:12px; font-size:0.88rem; }
footer { display:none !important; }
"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Road-Rail Resilience",
    theme=gr.themes.Soft(
        primary_hue="blue", neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=_CSS,
) as demo:

    gr.HTML("""
    <div class="header">
      <h2>Road-Rail Resilience: Early Warning Model</h2>
      <p>Kainos MSc Dissertation 2026
         &nbsp;|&nbsp; National Highways DATEX II + Network Rail TRUST
         &nbsp;|&nbsp; April 2026 &nbsp;|&nbsp; 10-25 km spatial filter</p>
    </div>
    <div class="legend">
      <span class="pill ptr">Training &nbsp;Apr 03-25</span>
      <span class="pill pte">Test &nbsp;Apr 25-28</span>
      <span class="pill ppr">Predicted &nbsp;Apr 09-30</span>
      <span style="color:#94a3b8; font-size:.72rem;">
        Apr 25 straddles train/test (row-based 80/20 split)
      </span>
    </div>
    """)

    with gr.Tabs():

        # ===================================================================
        # TAB A - By Closure
        # ===================================================================
        with gr.Tab("By Closure"):

            gr.HTML('<div class="sh">1. Filters</div>')
            with gr.Row():
                a_date     = gr.Textbox(label="Date (YYYY-MM-DD)",
                                        value="2026-04-11", scale=2)
                a_duration = gr.Slider(label="Max duration (h)",
                                       minimum=0, maximum=720, step=1,
                                       value=0, scale=3)
                a_distance = gr.Slider(label="Max station distance (km)",
                                       minimum=10, maximum=25, step=1,
                                       value=25, scale=3)
                a_btn      = gr.Button("Find Closures", variant="primary", scale=1)

            a_status = gr.Markdown("_Provide a date and click Find Closures._")

            gr.HTML('<div class="sh">2. Closures</div>')
            a_raw       = gr.State(pd.DataFrame())
            a_cl_tbl    = gr.Dataframe(interactive=False, wrap=True, max_height=220)

            gr.HTML('<div class="sh">3. Nearby stations</div>')
            with gr.Row():
                with gr.Column(scale=2):
                    a_nb_tbl  = gr.Dataframe(interactive=False, wrap=True,
                                             max_height=200)
                    a_nb_note = gr.Markdown("_Click a closure row._")
                with gr.Column(scale=1):
                    a_stn_dd  = gr.Dropdown(choices=[], label="Station",
                                            filterable=True,
                                            allow_custom_value=False)

            gr.HTML('<div class="sh">4. Predictions</div>')
            a_pred = gr.Markdown("_Select a station._", elem_classes=["pc"])

            gr.HTML('<div class="sh">5. Historical movements</div>')
            with gr.Group():
                with gr.Tabs():
                    with gr.Tab("Summary"):
                        a_wa_tbl = gr.Dataframe(interactive=False, wrap=True,
                                                max_height=280)
                    with gr.Tab("Delay trend"):
                        a_wa_fig = gr.Plot()

            gr.HTML('<div class="sh">6. Timetable predictions</div>')
            with gr.Group():
                with gr.Tabs():
                    with gr.Tab("Summary"):
                        a_wb_tbl = gr.Dataframe(interactive=False, wrap=True,
                                                max_height=280)
                    with gr.Tab("Disruption chart"):
                        a_wb_fig = gr.Plot()

            # Wiring
            a_btn.click(
                fn=find_closures,
                inputs=[a_date, a_duration, a_distance],
                outputs=[a_cl_tbl, a_status, a_raw],
            )
            a_cl_tbl.select(
                fn=closure_row_selected,
                inputs=[a_cl_tbl, a_raw, a_distance],
                outputs=[a_nb_tbl, a_nb_note, a_stn_dd],
            )
            a_stn_dd.change(
                fn=station_selected_a,
                inputs=[a_stn_dd, a_date],
                outputs=[a_pred, a_wa_tbl, a_wa_fig, a_wb_tbl, a_wb_fig],
            )

        # ===================================================================
        # TAB B - By Station
        # ===================================================================
        with gr.Tab("By Station"):

            gr.HTML('<div class="sh">1. Select station</div>')
            with gr.Row():
                b_stn_dd   = gr.Dropdown(choices=ALL_STATIONS, label="Station",
                                         filterable=True, allow_custom_value=False,
                                         scale=3)
                b_date     = gr.Textbox(label="Date (YYYY-MM-DD)",
                                        value="2026-04-11", scale=2)
                b_distance = gr.Slider(label="Max distance to closures (km)",
                                       minimum=10, maximum=25, step=1,
                                       value=25, scale=3)
                b_btn      = gr.Button("Load Station", variant="primary", scale=1)

            gr.HTML('<div class="sh">2. Closures near this station</div>')
            b_nb_note = gr.Markdown("_Select a station and click Load Station._")
            b_nb_tbl  = gr.Dataframe(interactive=False, wrap=True, max_height=200)

            gr.HTML('<div class="sh">3. Predictions</div>')
            b_pred = gr.Markdown("_Select a station._", elem_classes=["pc"])

            gr.HTML('<div class="sh">4. Historical movements</div>')
            with gr.Group():
                with gr.Tabs():
                    with gr.Tab("Summary"):
                        b_wa_tbl = gr.Dataframe(interactive=False, wrap=True,
                                                max_height=280)
                    with gr.Tab("Delay trend"):
                        b_wa_fig = gr.Plot()

            gr.HTML('<div class="sh">5. Timetable predictions</div>')
            with gr.Group():
                with gr.Tabs():
                    with gr.Tab("Summary"):
                        b_wb_tbl = gr.Dataframe(interactive=False, wrap=True,
                                                max_height=280)
                    with gr.Tab("Disruption chart"):
                        b_wb_fig = gr.Plot()

            # Wiring
            b_btn.click(
                fn=load_station,
                inputs=[b_stn_dd, b_date, b_distance],
                outputs=[
                    b_pred, b_nb_tbl,
                    b_wa_tbl, b_wa_fig,
                    b_wb_tbl, b_wb_fig,
                ],
            )
            b_btn.click(
                fn=lambda s, d: get_station_detail(s, parse_date(d))[0],
                inputs=[b_stn_dd, b_date],
                outputs=[b_nb_note],
            )
    with gr.Accordion("Feature Importance", open=False):
        gr.Markdown(
            "Gain-based feature importance from both trained models.  \n"
            "**Blue** = road closure features &nbsp;|&nbsp; "
            "**Orange** = temporal / volume features  \n"
        )
        with gr.Row():
            gr.Plot(value=_CLF_FIG, label="Classifier (XGBoost)")
            gr.Plot(value=_REG_FIG, label="Regression (XGBoost_tuned)")

    gr.Markdown(
        "_National Highways DATEX II · Network Rail TRUST · Darwin CIF · "
        " April 2026 · Spatial filter: 10-25 km_",
    )

if __name__ == "__main__":
    demo.launch(share=True, show_error=True, server_port=7860)