import json
import math
import os
import warnings
from datetime import date
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "notebooks", "data", "processed")
_MDL  = os.path.join(_ROOT, "notebooks", "models")

# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _candidate_paths(folder: str, stem: str) -> list[str]:
    """Return likely parquet/csv/json/pkl locations for a data or model artefact."""
    names = [stem]
    if "." not in os.path.basename(stem):
        names = [f"{stem}.parquet", f"{stem}.csv", f"{stem}.json", f"{stem}.pkl"]

    roots = [folder, _DATA, _MDL, _ROOT, os.getcwd(), "/mnt/data"]
    paths = []
    for root in roots:
        if not root:
            continue
        for name in names:
            path = os.path.join(root, name)
            if path not in paths:
                paths.append(path)
    return paths


def _first_existing(folder: str, stem: str) -> Optional[str]:
    """Find the first existing file for a named artefact."""
    for path in _candidate_paths(folder, stem):
        if os.path.exists(path):
            return path
    return None


def _read_table(stem: str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Load a processed table from parquet or csv, returning an empty frame if absent."""
    path = _first_existing(_DATA, stem)
    if path is None:
        print(f"  WARNING: processed file not found: {stem}")
        return pd.DataFrame(columns=columns or [])

    if path.endswith(".parquet"):
        available = pd.read_parquet(path).columns.tolist()
        use_cols = [c for c in (columns or available) if c in available]
        df = pd.read_parquet(path, columns=use_cols)
    elif path.endswith(".csv"):
        if columns:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            use_cols = [c for c in columns if c in header]
            df = pd.read_csv(path, usecols=use_cols)
        else:
            df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported table format: {path}")

    for col in columns or []:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _load_json(stem: str) -> dict:
    path = _first_existing(_MDL, stem)
    if path is None or not path.endswith(".json"):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"  WARNING: failed to read {path}: {exc}")
        return {}


def _load_model_bundle(stem: str, model_keys: tuple[str, ...]) -> tuple[dict, object | None]:
    """Load a joblib model bundle and return (metadata, trained_model)."""
    path = _first_existing(_MDL, stem)
    if path is None or not path.endswith(".pkl"):
        return {}, None
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        print(f"  WARNING: failed to load {path}: {exc}")
        return {}, None

    if isinstance(bundle, dict):
        meta = bundle.get("meta", {}) or {}
        model = None
        for key in model_keys:
            if key in bundle:
                model = bundle[key]
                break
        return meta, model
    return {}, bundle


def _feature_importance_from_model(model, features: list[str]) -> list[tuple[str, float]]:
    """Extract normalised feature importance from an XGBoost model if available."""
    if model is None:
        return []

    try:
        booster = model.get_booster()
        raw = booster.get_score(importance_type="gain")
        if raw:
            if booster.feature_names:
                scores = {name: float(raw.get(name, 0.0)) for name in booster.feature_names}
            else:
                scores = {features[int(k[1:])]: float(v) for k, v in raw.items() if k.startswith("f") and int(k[1:]) < len(features)}
            total = sum(scores.values()) or 1.0
            return sorted(((k, v / total) for k, v in scores.items()), key=lambda x: x[1], reverse=True)
    except Exception:
        pass

    try:
        values = np.asarray(model.feature_importances_, dtype=float)
        if len(values) and len(features) == len(values):
            total = float(values.sum()) or 1.0
            return sorted(zip(features, (values / total).tolist()), key=lambda x: x[1], reverse=True)
    except Exception:
        pass

    return []


def _derive_train_test_dates(df: pd.DataFrame, train_rows: Optional[int]) -> tuple[date, date, date]:
    """Infer train/test date windows from station-day rows and model metadata."""
    if df.empty or "planned_date" not in df.columns:
        today = date.today()
        return today, today, today

    dates = pd.to_datetime(df["planned_date"], errors="coerce").dropna().dt.date
    if dates.empty:
        today = date.today()
        return today, today, today

    train_start = dates.min()
    test_end = dates.max()

    if train_rows:
        counts = pd.Series(dates).value_counts().sort_index()
        cum_counts = counts.cumsum()
        train_candidates = cum_counts[cum_counts <= int(train_rows)]
        if not train_candidates.empty:
            train_end = train_candidates.index[-1]
        else:
            train_end = train_start
    else:
        train_end = train_start

    return train_start, train_end, test_end


def _derive_prediction_dates(df: pd.DataFrame) -> tuple[date, date]:
    if df.empty or "planned_date" not in df.columns:
        today = date.today()
        return today, today
    dates = pd.to_datetime(df["planned_date"], errors="coerce").dropna().dt.date
    if dates.empty:
        today = date.today()
        return today, today
    return dates.min(), dates.max()

# ---------------------------------------------------------------------------
# Model metadata and dynamic feature importance
# ---------------------------------------------------------------------------

print("Loading model metadata...")
clf_meta: dict = _load_json("road_rail_classification_meta")
_clf_pkl_meta, clf_model = _load_model_bundle("road_rail_classification_model", ("pipeline", "model"))
clf_meta.update({k: v for k, v in _clf_pkl_meta.items() if k not in clf_meta})

reg_meta, reg_model = _load_model_bundle("road_rail_regression_model", ("model", "pipeline"))

CLF_IMP = _feature_importance_from_model(clf_model, clf_meta.get("features", [])) 
REG_IMP = _feature_importance_from_model(reg_model, reg_meta.get("features", []))

_ROAD_FEATURES = {
    "has_road_closure", "road_closure_count", "n_unplanned_closures",
    "min_distance_km", "mean_distance_km", "max_effective_duration_hours",
    "mean_effective_duration_hours", "inv_distance_sum",
    "total_closure_severity", "max_road_class",
    "closures_lag1d", "closures_lag3d", "closures_lag7d",
}

OPT_THRESHOLD = float(clf_meta.get("optimal_threshold", 0.5))

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

print("Loading closures...")
_closure_cols = [
    "situation_id", "road_name", "closure_type", "cause_type",
    "start_time_dt", "end_time_dt", "lanes_closed", "duration_hours",
    "closure_lat", "closure_lon",
]
closures_df = _read_table("road_closures_clean", _closure_cols)
for col in _closure_cols:
    if col not in closures_df.columns:
        closures_df[col] = pd.NA
closures_df["start_time_dt"] = pd.to_datetime(closures_df["start_time_dt"], utc=True, errors="coerce")
closures_df["end_time_dt"] = pd.to_datetime(closures_df["end_time_dt"], utc=True, errors="coerce")
closures_df["situation_id"] = closures_df["situation_id"].astype(str)
closures_df["duration_hours"] = pd.to_numeric(closures_df["duration_hours"], errors="coerce").fillna(0)

print("Loading station metrics...")
_station_cols = [
    "station_name", "stanox", "planned_date", "train_movements",
    "mean_delay_minutes", "late_share", "delayed_5min_share",
    "severe_15min_share", "road_closure_count",
    "n_unplanned_closures", "min_distance_km", "has_road_closure",
]
station_day_df = _read_table("road_station_day_dataset", _station_cols)
station_day_df["planned_date"] = pd.to_datetime(station_day_df["planned_date"], errors="coerce").dt.date

print("Loading timetable predictions...")
_tt_want = [
    "station_name", "stanox", "planned_date", "train_movements",
    "disruption_probability", "disrupted_predicted", "risk_band",
    "predicted_delay_minutes",
]
timetable_df = _read_table("road_timetable_station_day", _tt_want)
timetable_df["planned_date"] = pd.to_datetime(timetable_df["planned_date"], errors="coerce").dt.date
for _col, _default in [
    ("disruption_probability", 0.0),
    ("disrupted_predicted",    False),
    ("risk_band",              "N/A"),
    ("predicted_delay_minutes", 0.0),
]:
    if _col not in timetable_df.columns:
        timetable_df[_col] = _default

timetable_df["disruption_probability"] = pd.to_numeric(timetable_df["disruption_probability"], errors="coerce").fillna(0.0)
timetable_df["predicted_delay_minutes"] = pd.to_numeric(timetable_df["predicted_delay_minutes"], errors="coerce").fillna(0.0)

print("Loading station reference...")
_station_ref_path = _first_existing(_DATA, "stations_reference")
if _station_ref_path:
    _raw_stations = _read_table("stations_reference")
    stations_ref = _raw_stations.rename(columns={"station": "station_name"}).copy()
else:
    print("  WARNING: station reference file not found. Map station coordinates will be unavailable.")
    stations_ref = pd.DataFrame(columns=["station_name", "stanox", "latitude", "longitude"])
for col in ["station_name", "stanox", "latitude", "longitude"]:
    if col not in stations_ref.columns:
        stations_ref[col] = pd.NA
stations_ref = stations_ref.dropna(subset=["latitude", "longitude"])

ALL_STATIONS = sorted(station_day_df["station_name"].dropna().unique().tolist())

TRAIN_START, TRAIN_END, TEST_END = _derive_train_test_dates(station_day_df, clf_meta.get("train_rows"))
PRED_START, PRED_END = _derive_prediction_dates(timetable_df)


print(f"Ready - closures: {len(closures_df):,}  stations: {len(stations_ref):,}")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def history_fig(hist: pd.DataFrame, station: str, train_start, train_end, test_end) -> plt.Figure:
    _BG    = "#f9fafb"
    dates  = [pd.Timestamp(d) for d in hist["planned_date"]]
    delays = hist["mean_delay_minutes"].values
    late   = hist["late_share"].values * 100

    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.axvspan(pd.Timestamp(train_start), pd.Timestamp(train_end), alpha=0.07, color="#3b82f6")
        ax.axvspan(pd.Timestamp(train_end),   pd.Timestamp(test_end),  alpha=0.12, color="#f97316")

    axes[0].fill_between(dates, delays, alpha=0.12, color="#ef4444")
    axes[0].plot(dates, delays, "o-", color="#ef4444", lw=1.8, ms=3.5)
    axes[0].axhline(0, color="#cbd5e1", lw=1, linestyle="--")
    axes[0].axhline(5, color="#ef4444", lw=1.2, linestyle=":", label="5 min threshold")
    axes[0].set_ylabel("Mean delay (min)", fontsize=8)
    axes[0].set_title(f"Historical - {station}", fontsize=9, fontweight="bold")
    axes[0].legend(
        handles=axes[0].get_legend_handles_labels()[0] + [
            mpatches.Patch(color="#3b82f6", alpha=0.3, label="Training"),
            mpatches.Patch(color="#f97316", alpha=0.4, label="Test"),
        ],
        fontsize=7, loc="upper left", ncol=3,
    )
    axes[1].bar(
        dates, late,
        color=["#ef4444" if v > 50 else "#3b82f6" for v in late],
        alpha=0.65, width=0.8,
    )
    axes[1].set_ylabel("% late", fontsize=8)
    axes[1].set_ylim(0, max(100, late.max() * 1.15) if len(late) else 100)
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout(h_pad=0.4)
    return fig


def timetable_fig(tt: pd.DataFrame, station: str, pred_start, pred_end, threshold: float) -> plt.Figure:
    _BG   = "#f9fafb"
    dates = [pd.Timestamp(d) for d in tt["planned_date"]]
    svcs  = tt["train_movements"].values
    probs = tt["disruption_probability"].values * 100
    delay = tt["predicted_delay_minutes"].values
    pc    = "#8b5cf6"

    fig, axes = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True)
    fig.patch.set_facecolor(_BG)
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.axvspan(pd.Timestamp(pred_start), pd.Timestamp(pred_end), alpha=0.05, color=pc)

    axes[0].bar(dates, svcs, color="#3b82f6", alpha=0.6, width=0.8)
    axes[0].set_ylabel("Scheduled", fontsize=8)
    axes[0].set_title(f"Timetable predictions - {station}", fontsize=9, fontweight="bold")
    axes[0].legend(handles=[mpatches.Patch(color=pc, alpha=0.2, label="Prediction window")], fontsize=7, loc="upper left")

    axes[1].fill_between(dates, probs, alpha=0.15, color=pc)
    axes[1].plot(dates, probs, "o-", color=pc, lw=1.8, ms=3.5)
    axes[1].axhline(threshold * 100, color="#ef4444", linestyle=":", lw=1.2, label=f"Threshold {threshold:.0%}")
    axes[1].set_ylabel("Disruption prob (%)", fontsize=8)
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=7, loc="upper left")

    axes[2].bar(dates, delay, color=["#ef4444" if v > 0 else "#22c55e" for v in delay], alpha=0.75, width=0.8)
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


def imp_fig(imp_list: list, title: str) -> plt.Figure:
    _BG    = "#f9fafb"
    feats  = [f for f, _ in imp_list]
    vals   = [v for _, v in imp_list]
    colors = ["#3b82f6" if f in _ROAD_FEATURES else "#f97316" for f in feats]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.barh(feats[::-1], vals[::-1], color=colors[::-1], alpha=0.82, edgecolor="white")
    for i, (_, v) in enumerate(zip(feats[::-1], vals[::-1])):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlabel("Importance (gain)", fontsize=8)
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.legend(
        handles=[
            mpatches.Patch(color="#3b82f6", alpha=0.82, label="Road closure feature"),
            mpatches.Patch(color="#f97316", alpha=0.82, label="Temporal / volume feature"),
        ],
        fontsize=7, loc="lower right",
    )
    plt.tight_layout()
    return fig


CLF_FIG = imp_fig(CLF_IMP, "Classifier - XGBoost")
REG_FIG = imp_fig(REG_IMP, "Regression - XGBoost_regularised")

# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def haversine(la1: float, lo1: float, la2: float, lo2: float) -> float:
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [la1, lo1, la2, lo2])
    dla, dlo = la2 - la1, lo2 - lo1
    a = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

# ---------------------------------------------------------------------------
# Date / time parsing
# ---------------------------------------------------------------------------

def parse_date(datetime_str: str) -> date:
    s = str(datetime_str or "").strip().replace(" ", "T")[:10]
    try:
        return date.fromisoformat(s)
    except Exception:
        return TRAIN_START


def parse_datetime(datetime_str: str) -> Optional[pd.Timestamp]:
    s = str(datetime_str or "").strip().replace(" ", "T")
    if "T" not in s or len(s) < 13:
        return None
    time_part = s[11:16]
    if time_part in ("", "00:00"):
        return None
    try:
        return pd.Timestamp(s[:16] + ":00+00:00", tz="UTC")
    except Exception:
        return None


def combine_datetime(date_str: str, time_str: str) -> str:
    d = str(date_str or "").strip()
    t = str(time_str or "").strip()
    return f"{d}T{t}" if t and t != "00:00" else d

# ---------------------------------------------------------------------------
# Closure queries
# ---------------------------------------------------------------------------

def _active_closures(ts: Optional[pd.Timestamp], d: date) -> pd.DataFrame:
    if ts is not None:
        start = pd.to_datetime(closures_df["start_time_dt"], utc=True, errors="coerce")
        end   = pd.to_datetime(closures_df["end_time_dt"],   utc=True, errors="coerce")
        mask  = (start <= ts) & ((end >= ts) | end.isna())
        return closures_df[mask].copy()
    return closures_df[closures_df["start_time_dt"].dt.date == d].copy()


def _fmt_closure_df(ac: pd.DataFrame) -> pd.DataFrame:
    disp = ac[[
        "situation_id", "road_name", "closure_type", "cause_type",
        "duration_hours", "start_time_dt", "end_time_dt",
    ]].copy()
    disp.columns = ["ID", "Road", "Type", "Cause", "Dur (h)", "Start (UTC)", "End (UTC)"]
    disp["Dur (h)"]     = disp["Dur (h)"].round(1)
    disp["Start (UTC)"] = pd.to_datetime(disp["Start (UTC)"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    disp["End (UTC)"]   = pd.to_datetime(disp["End (UTC)"],   utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    return disp


def _time_label(datetime_str: str, ts: Optional[pd.Timestamp]) -> str:
    if ts is None:
        return ""
    s = str(datetime_str or "").strip().replace(" ", "T")
    return f" at {s[11:16]} UTC" if len(s) >= 16 else ""


def get_closures(datetime_str: str, ctype: str, max_dist: float, max_dur: float):
    d  = parse_date(datetime_str)
    ts = parse_datetime(datetime_str)
    tl = _time_label(datetime_str, ts)

    ac = _active_closures(ts, d)
    if ctype == "Planned":     ac = ac[ac["closure_type"] == "planned"]
    elif ctype == "Unplanned": ac = ac[ac["closure_type"] == "unplanned"]
    ac = ac[ac["duration_hours"] <= float(max_dur or 72)].dropna(subset=["closure_lat", "closure_lon"])

    if ac.empty:
        return pd.DataFrame(), f"No active closures on {d}{tl}.", [], ts

    p, u  = (ac["closure_type"] == "planned").sum(), (ac["closure_type"] == "unplanned").sum()
    disp  = _fmt_closure_df(ac).sort_values("Start (UTC)").reset_index(drop=True)
    label = f"<b>{len(ac)}</b> active closure(s) on {d}{tl} &nbsp;·&nbsp; {p} planned &nbsp;·&nbsp; {u} unplanned"
    return disp, label, ac.to_dict("records"), ts


def get_closures_near_station(stn: str, datetime_str: str, ctype: str, max_dist: float, max_dur: float):
    d  = parse_date(datetime_str)
    ts = parse_datetime(datetime_str)
    tl = _time_label(datetime_str, ts)

    sr = stations_ref[stations_ref["station_name"] == stn]
    if sr.empty:
        return pd.DataFrame(), f"'{stn}' not in spatial reference.", [], ts

    slat = float(sr.iloc[0]["latitude"])
    slon = float(sr.iloc[0]["longitude"])

    ac = _active_closures(ts, d)
    if ctype == "Planned":     ac = ac[ac["closure_type"] == "planned"]
    elif ctype == "Unplanned": ac = ac[ac["closure_type"] == "unplanned"]
    ac = ac[ac["duration_hours"] <= float(max_dur or 72)]

    rows = []
    for _, cl in ac.iterrows():
        if pd.isna(cl.get("closure_lat")) or pd.isna(cl.get("closure_lon")):
            continue
        dist = haversine(slat, slon, cl["closure_lat"], cl["closure_lon"])
        if dist <= float(max_dist or 25):
            rows.append({**cl.to_dict(), "distance_km": round(dist, 1)})

    if not rows:
        return pd.DataFrame(), f"No active closures within {max_dist:.0f} km of {stn} on {d}{tl}.", [], ts

    raw   = pd.DataFrame(rows).sort_values("distance_km").reset_index(drop=True)
    p, u  = (raw["closure_type"] == "planned").sum(), (raw["closure_type"] == "unplanned").sum()
    disp  = _fmt_closure_df(raw).reset_index(drop=True)
    label = (f"<b>{len(raw)}</b> active closure(s) within {max_dist:.0f} km of <b>{stn}</b>"
             f" on {d}{tl} &nbsp;·&nbsp; {p} planned &nbsp;·&nbsp; {u} unplanned")
    return disp, label, raw.to_dict("records"), ts


def get_nearby_stations(closure_df: pd.DataFrame, max_km: float = 25.0) -> pd.DataFrame:
    rows = []
    for _, cl in closure_df.iterrows():
        if pd.isna(cl.get("closure_lat")) or pd.isna(cl.get("closure_lon")):
            continue
        for _, st in stations_ref.iterrows():
            d = haversine(cl["closure_lat"], cl["closure_lon"], st["latitude"], st["longitude"])
            if d <= max_km:
                rows.append({
                    "station_name": st["station_name"],
                    "distance_km":  round(d, 1),
                    "closure_id":   cl["situation_id"],
                    "road":         cl.get("road_name", ""),
                    "type":         cl.get("closure_type", ""),
                })
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("distance_km")
        .drop_duplicates("station_name")
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------------
# Station risk lookup
# ---------------------------------------------------------------------------

def get_station_risk(stn: str, d: date) -> dict:
    row = timetable_df[
        (timetable_df["station_name"] == stn) & (timetable_df["planned_date"] == d)
    ]
    if row.empty:
        return {"prob": 0.0, "risk": "N/A", "disrupted": False, "delay": 0.0}
    r = row.iloc[0]
    return {
        "prob":      float(r.get("disruption_probability", 0)),
        "risk":      str(r.get("risk_band", "N/A")),
        "disrupted": bool(r.get("disrupted_predicted", False)),
        "delay":     float(r.get("predicted_delay_minutes", 0)),
    }

# ---------------------------------------------------------------------------
# Station panels
# ---------------------------------------------------------------------------

def get_station_panels(stn: str, d: date):
    risk = get_station_risk(stn, d)

    colour    = {"high": "#d4351c", "medium": "#f47738", "low": "#00703c"}.get(risk["risk"].lower(), "#505a5f")
    direction = "early" if risk["delay"] < -0.5 else "late" if risk["delay"] > 0.5 else "on time"
    disruption_label = "Disruption expected" if risk["disrupted"] else "No disruption expected"
    badge = (
        f"<div style='font-family:Arial'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
        f"<span style='background:{colour};color:#fff;font-size:11px;font-weight:700;"
        f"text-transform:uppercase;padding:3px 10px'>{risk['risk'].upper()}</span>"
        f"<span style='font-size:0.82rem;color:#505a5f'>{disruption_label}</span></div>"
        f"<table style='border-collapse:collapse;width:100%;font-size:0.83rem'>"
        f"<tr><td style='padding:5px 6px;color:#505a5f'>Disruption probability</td>"
        f"    <td style='padding:5px 6px;font-weight:700;color:#1a3a5c'>{risk['prob']:.0%}</td></tr>"
        f"<tr><td style='padding:5px 6px;color:#505a5f'>Predicted delay</td>"
        f"    <td style='padding:5px 6px;font-weight:700;color:#1a3a5c'>"
        f"    {risk['delay']:+.1f} min ({direction})</td></tr>"
        f"</table></div>"
    )

    hist = station_day_df[station_day_df["station_name"] == stn].sort_values("planned_date").copy()
    if not hist.empty:
        wa = hist[["planned_date", "train_movements", "mean_delay_minutes", "late_share", "delayed_5min_share", "severe_15min_share"]].copy()
        wa.columns = ["Date", "Services", "Mean delay (min)", "% late", "% >5 min", "% >15 min"]
        wa["Mean delay (min)"] = wa["Mean delay (min)"].round(1)
        for c in ["% late", "% >5 min", "% >15 min"]:
            wa[c] = (wa[c] * 100).round(1)
        wf = history_fig(hist, stn, TRAIN_START, TRAIN_END, TEST_END)
    else:
        wa, wf = pd.DataFrame(), None

    tt = timetable_df[timetable_df["station_name"] == stn].sort_values("planned_date").copy()
    if not tt.empty:
        wb = tt[["planned_date", "train_movements", "disruption_probability", "risk_band", "predicted_delay_minutes"]].copy()
        wb.columns = ["Date", "Services", "Probability", "Risk", "Delay (min)"]
        wb["Probability"] = (wb["Probability"] * 100).round(1).astype(str) + "%"
        wb["Delay (min)"] = wb["Delay (min)"].round(1)
        tf = timetable_fig(tt, stn, PRED_START, PRED_END, OPT_THRESHOLD)
    else:
        wb, tf = pd.DataFrame(), None

    return badge, wa, wf, wb, tf