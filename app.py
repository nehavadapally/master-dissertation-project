"""
Road-Rail Resilience: Early Warning Demo
Kainos MSc Dissertation 2026

Column name mapping (confirmed from EDA notebooks):
    road_closures_clean        -> join key: situation_id  (numeric string e.g. "481398")
    road_timetable_dataset     -> join key: closure_id    (== situation_id)
    road_train_moments_dataset -> join key: closure_id    (== situation_id)

Flow:
  1. User picks date + time
  2. Filter road_closures_clean  : start_time_dt <= dt <= end_time_dt
  3. User clicks a closure row   -> situation_id selected
  4. Filter road_timetable_dataset    by closure_id == situation_id -> run model
  5. Filter road_train_moments_dataset by closure_id == situation_id -> actual delays
  6. Show predictions table + historical table + gauge
"""

import os
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "notebooks/data", "processed")
MODEL_PATH     = os.path.join(BASE_DIR, "notebooks/models", "road_rail_model.pkl")
CLOSURES_PATH  = os.path.join(DATA_DIR, "road_closures_clean.parquet")
TIMETABLE_PATH = os.path.join(DATA_DIR, "road_timetable_dataset.parquet")
MOMENTS_PATH   = os.path.join(DATA_DIR, "road_train_moments_dataset.parquet")

# ── Load parquets once at startup ─────────────────────────────────────────────
print("Loading parquet files...")

# road_closures_clean: join key is "situation_id" (numeric string e.g. "481398")
closures_df = pd.read_parquet(CLOSURES_PATH, columns=[
    "situation_id", "road_name", "closure_type", "cause_type",
    "start_time_dt", "end_time_dt", "lanes_closed", "duration_hours",
])
closures_df["start_time_dt"] = pd.to_datetime(closures_df["start_time_dt"], utc=True)
closures_df["end_time_dt"]   = pd.to_datetime(closures_df["end_time_dt"],   utc=True)
print(f"  closures  : {len(closures_df):,} rows")

# road_timetable_dataset: join key is "closure_id" (== situation_id)
# confirmed columns: closure_id, closure_type, station_name, distance_in_km,
#   closure_start_time, planned_timestamp, planned_time_diff, event_type,
#   distance_time_interaction, hour, time_bucket, stanox, tpl, ...
timetable_df = pd.read_parquet(TIMETABLE_PATH, columns=[
    "closure_id", "closure_type", "station_name", "distance_in_km",
    "planned_timestamp", "planned_time_diff",
    "event_type", "distance_time_interaction",
])
timetable_df["planned_timestamp"] = pd.to_datetime(
    timetable_df["planned_timestamp"], utc=True
)
print(f"  timetable : {len(timetable_df):,} rows")

# road_train_moments_dataset: join key is "closure_id" (== situation_id)
# confirmed columns: closure_id, closure_type, station_name, distance_in_km,
#   closure_start_time, actual_timestamp, planned_timestamp, planned_time_diff,
#   event_type, variation_status, delay, distance_time_interaction, ...
moments_df = pd.read_parquet(MOMENTS_PATH, columns=[
    "closure_id", "station_name", "distance_in_km",
    "planned_timestamp", "event_type", "delay",
])
moments_df["planned_timestamp"] = pd.to_datetime(
    moments_df["planned_timestamp"], utc=True
)
print(f"  moments   : {len(moments_df):,} rows")

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    if os.path.exists(MODEL_PATH):
        mdl = joblib.load(MODEL_PATH)
        print(f"  model     : loaded")
        return mdl, False
    print(f"  WARNING   : model not found - mock active")
    return None, True

pipeline, IS_MOCK = load_model()

# Must match modelling.ipynb training exactly
FEATURES = [
    "distance_in_km", "planned_time_diff", "closure_type",
    "event_type", "distance_time_interaction",
]

class MockPipeline:
    """Mirrors real model coefficients from notebook output."""
    def predict(self, X):
        base = 1.2
        base += np.where(X["event_type"]   == "ARRIVAL",  0.055, -0.055)
        base += np.where(X["closure_type"] == "planned",  0.054, -0.054)
        base += -0.025 * X["distance_in_km"].values
        base += 0.0003 * X["planned_time_diff"].values
        return np.maximum(0, base + np.random.uniform(-0.1, 0.1, len(X)))

# ── Risk band ─────────────────────────────────────────────────────────────────
def risk_band(delay):
    if delay < 1:  return "Minimal"
    if delay < 3:  return "Low"
    if delay < 5:  return "Moderate"
    return "High"

# ── Gauge chart ───────────────────────────────────────────────────────────────
def gauge_figure(delay_min):
    colours = {"Minimal": "#2ecc71", "Low": "#f1c40f",
               "Moderate": "#e67e22", "High": "#e74c3c"}
    colour = colours[risk_band(delay_min)]
    fig, ax = plt.subplots(figsize=(5, 1.2))
    ax.barh([0], [10],                 color="#ecf0f1", height=0.5)
    ax.barh([0], [min(delay_min, 10)], color=colour,   height=0.5)
    ax.set_xlim(0, 10)
    ax.set_yticks([])
    ax.set_xlabel("Predicted delay (min)")
    ax.set_title(f"{delay_min:.2f} min  ({risk_band(delay_min)})",
                 fontweight="bold", fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    return fig

# ── Step 1->2: filter closures by chosen datetime ─────────────────────────────
def get_active_closures(chosen_dt):
    if not chosen_dt:
        return pd.DataFrame(), "Pick a date and time above."

    # make timezone-aware so it compares correctly with UTC parquet timestamps
    dt = pd.to_datetime(chosen_dt, utc=True)

    active = closures_df[
        (closures_df["start_time_dt"] <= dt) &
        (closures_df["end_time_dt"]   >= dt)
    ].copy()

    if active.empty:
        return pd.DataFrame(), (
            f"No active closures at {dt.strftime('%Y-%m-%d %H:%M UTC')}. "
            f"Try a date between {closures_df['start_time_dt'].min().date()} "
            f"and {closures_df['end_time_dt'].max().date()}."
        )

    display = active[[
        "situation_id", "road_name", "closure_type", "cause_type",
        "lanes_closed", "duration_hours", "start_time_dt", "end_time_dt",
    ]].rename(columns={
        "start_time_dt":  "starts",
        "end_time_dt":    "ends",
        "duration_hours": "duration_h",
    }).reset_index(drop=True)

    msg = (
        f"Found {len(display)} active closure(s) at "
        f"{dt.strftime('%Y-%m-%d %H:%M UTC')}. "
        f"Click a row to see predictions."
    )
    return display, msg

# ── Step 3->5: row click -> predictions ───────────────────────────────────────
def on_closure_selected(closures_display, selected: gr.SelectData):
    if closures_display is None or closures_display.empty:
        return pd.DataFrame(), pd.DataFrame(), None, "No closure selected."

    row   = closures_display.iloc[selected.index[0]]
    sid   = row["situation_id"]   # join key — matches closure_id in joined datasets
    road  = row["road_name"]
    ctype = row["closure_type"]

    # lookup in joined datasets by closure_id (== situation_id)
    tt   = timetable_df[timetable_df["closure_id"] == sid].copy()
    hist = moments_df[moments_df["closure_id"]     == sid].copy()

    if tt.empty:
        return (
            pd.DataFrame(), pd.DataFrame(), None,
            f"No timetable data found for situation_id={sid}.\n"
            f"Check that closure_id in road_timetable_dataset matches situation_id."
        )

    # run model on all timetable rows at once
    mdl = pipeline if not IS_MOCK else MockPipeline()
    tt["predicted_delay"] = mdl.predict(tt[FEATURES])
    tt["risk"]            = tt["predicted_delay"].apply(risk_band)

    pred_table = (
        tt[["station_name", "planned_timestamp", "event_type",
            "distance_in_km", "predicted_delay", "risk"]]
        .rename(columns={
            "station_name":      "station",
            "planned_timestamp": "planned_time",
            "event_type":        "event",
            "distance_in_km":    "dist_km",
            "predicted_delay":   "pred_delay_min",
        })
        .sort_values("planned_time")
        .reset_index(drop=True)
    )
    pred_table["pred_delay_min"] = pred_table["pred_delay_min"].round(2)
    pred_table["dist_km"]        = pred_table["dist_km"].round(2)

    # historical actual delays
    hist_table = pd.DataFrame()
    if not hist.empty:
        hist_table = (
            hist[["station_name", "planned_timestamp", "event_type",
                  "distance_in_km", "delay"]]
            .rename(columns={
                "station_name":      "station",
                "planned_timestamp": "planned_time",
                "event_type":        "event",
                "distance_in_km":    "dist_km",
                "delay":             "actual_delay_min",
            })
            .sort_values("planned_time")
            .reset_index(drop=True)
        )
        hist_table["actual_delay_min"] = hist_table["actual_delay_min"].round(2)
        hist_table["dist_km"]          = hist_table["dist_km"].round(2)

    worst = float(tt["predicted_delay"].max())
    gauge = gauge_figure(worst)

    summary = (
        f"Situation ID : {sid}\n"
        f"Road         : {road}  |  Type: {ctype}\n"
        f"Scheduled    : {len(pred_table):,} train services affected\n"
        f"Historical   : {len(hist_table):,} actual train records\n"
        f"Worst predicted delay : {worst:.2f} min  ({risk_band(worst)})"
        + ("\n  Mock model active." if IS_MOCK else "")
    )

    return pred_table, hist_table, gauge, summary

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Road-Rail Early Warning") as demo:

    gr.Markdown("""
    # Road-Rail Resilience - Early Warning Model Demo
    *Kainos MSc Dissertation 2026*

    **How to use:** Pick a date/time -> click Find -> click any closure row -> predictions appear.
    """)

    with gr.Row():
        dt_input   = gr.DateTime(
            label="Select date and time (UTC)",
            value="2026-04-11 21:00:00",
            type="string",
        )
        search_btn = gr.Button("Find active closures", variant="primary", scale=0)

    status_box = gr.Textbox(label="Status", interactive=False, lines=1)

    gr.Markdown("### Active road closures at selected time")
    closures_table = gr.Dataframe(
        label="Click a row to load predictions",
        interactive=False,
        wrap=True,
    )

    gr.Markdown("### Predictions & historical data")
    with gr.Row():
        with gr.Column():
            summary_box = gr.Textbox(label="Summary", lines=7, interactive=False)
            gauge_plot  = gr.Plot(label="Worst-case delay gauge")
        with gr.Column():
            pred_table  = gr.Dataframe(
                label="Predicted delays (from timetable)", interactive=False
            )

    gr.Markdown("### Historical actual delays (same closure, from train moments)")
    hist_table = gr.Dataframe(label="Actual recorded delays", interactive=False)

    # Wiring
    search_btn.click(
        fn=get_active_closures,
        inputs=[dt_input],
        outputs=[closures_table, status_box],
    )
    closures_table.select(
        fn=on_closure_selected,
        inputs=[closures_table],
        outputs=[pred_table, hist_table, gauge_plot, summary_box],
    )

if __name__ == "__main__":
    demo.launch(share=True, show_error=True, server_port=7860)