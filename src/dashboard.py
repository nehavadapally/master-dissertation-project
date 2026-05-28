import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import gradio as gr

import src.data as D

CLF_FIG = D.CLF_FIG
REG_FIG = D.REG_FIG

_BG = "#f9fafb"

# ---------------------------------------------------------------------------
# Dynamic metrics loaded from processed data and model metadata
# ---------------------------------------------------------------------------

# The dashboard should report the current processed outputs, not copied values.
# Metrics below are derived from src.data, which loads the processed station-day
# datasets and model metadata at application start-up.

def _date_series(df: pd.DataFrame, col: str = "planned_date") -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df[col], errors="coerce").dropna()


def _date_label(start, end) -> str:
    if pd.isna(start) or pd.isna(end):
        return "Date range unavailable"
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s.year == e.year:
        return f"{s.day} {s.strftime('%b')} - {e.day} {e.strftime('%b')} {e.year}"
    return f"{s.day} {s.strftime('%b')} {s.year} - {e.day} {e.strftime('%b')} {e.year}"


def _split_ranges(df: pd.DataFrame, train_rows: int, test_rows: int) -> dict:
    dates = _date_series(df)
    if dates.empty:
        return {
            "dataset_start": pd.NaT, "dataset_end": pd.NaT,
            "train_start": pd.NaT, "train_end": pd.NaT,
            "test_start": pd.NaT, "test_end": pd.NaT,
        }

    by_date = dates.dt.date.value_counts().sort_index()
    if by_date.empty:
        return {
            "dataset_start": dates.min(), "dataset_end": dates.max(),
            "train_start": dates.min(), "train_end": dates.max(),
            "test_start": dates.min(), "test_end": dates.max(),
        }

    csum = by_date.cumsum()
    train_end_candidates = csum[csum >= int(train_rows or 0)]
    train_end = train_end_candidates.index[0] if len(train_end_candidates) else by_date.index[-1]

    train_dates = [d for d in by_date.index if d <= train_end]
    test_dates = [d for d in by_date.index if d > train_end]

    return {
        "dataset_start": by_date.index[0],
        "dataset_end": by_date.index[-1],
        "train_start": train_dates[0] if train_dates else by_date.index[0],
        "train_end": train_dates[-1] if train_dates else train_end,
        "test_start": test_dates[0] if test_dates else train_end,
        "test_end": test_dates[-1] if test_dates else by_date.index[-1],
    }


def _target_column(df: pd.DataFrame) -> str | None:
    for col in ("station_disrupted", "disrupted", "target"):
        if col in df.columns:
            return col
    if "mean_delay_minutes" in df.columns:
        return "mean_delay_minutes"
    return None


def _build_clf_metrics() -> dict:
    meta = getattr(D, "clf_meta", {}) or {}
    station_df = getattr(D, "station_day_df", pd.DataFrame())
    timetable_df = getattr(D, "timetable_df", pd.DataFrame())

    train_rows = int(meta.get("train_rows", 0) or 0)
    test_rows = int(meta.get("test_rows", 0) or 0)
    if not train_rows and not station_df.empty:
        train_rows = len(station_df)

    ranges = _split_ranges(station_df, train_rows, test_rows)
    total_station_days = int(len(station_df)) if station_df is not None else 0
    unique_stations = int(station_df["station_name"].nunique()) if "station_name" in station_df.columns else 0

    threshold = float(meta.get("disruption_threshold_min", 5) or 5)
    target = _target_column(station_df)
    if target == "mean_delay_minutes":
        disrupted = int((pd.to_numeric(station_df[target], errors="coerce") > threshold).sum())
    elif target:
        disrupted = int(pd.to_numeric(station_df[target], errors="coerce").fillna(0).astype(int).sum())
    else:
        disrupted = 0
    not_disrupted = max(total_station_days - disrupted, 0)
    disruption_rate = disrupted / total_station_days if total_station_days else float(meta.get("test_disruption_rate", 0) or 0)
    class_imbalance = not_disrupted / disrupted if disrupted else 0

    pred_total = int(len(timetable_df)) if timetable_df is not None else 0
    risk_counts = {"low": 0, "moderate": 0, "high": 0, "critical": 0}
    if timetable_df is not None and not timetable_df.empty and "risk_band" in timetable_df.columns:
        counts = timetable_df["risk_band"].fillna("N/A").astype(str).str.lower().value_counts()
        for key in risk_counts:
            risk_counts[key] = int(counts.get(key, 0))

    pred_dates = _date_series(timetable_df)

    return {
        "model_name": meta.get("model_name", "Classifier"),
        "pr_auc": float(meta.get("pr_auc", 0) or 0),
        "roc_auc": float(meta.get("roc_auc", 0) or 0),
        "f1": float(meta.get("f1", 0) or 0),
        "optimal_threshold": float(meta.get("optimal_threshold", getattr(D, "OPT_THRESHOLD", 0)) or 0),
        "disruption_threshold_min": threshold,
        "total_station_days": total_station_days,
        "train_rows": train_rows,
        "test_rows": int(test_rows or max(total_station_days - train_rows, 0)),
        "unique_stations": unique_stations,
        "not_disrupted": not_disrupted,
        "disrupted": disrupted,
        "disruption_rate": disruption_rate,
        "class_imbalance": class_imbalance,
        "pred_total": pred_total,
        "risk_low": risk_counts["low"],
        "risk_moderate": risk_counts["moderate"],
        "risk_high": risk_counts["high"],
        "risk_critical": risk_counts["critical"],
        "dataset_start": ranges["dataset_start"],
        "dataset_end": ranges["dataset_end"],
        "train_start": ranges["train_start"],
        "train_end": ranges["train_end"],
        "test_start": ranges["test_start"],
        "test_end": ranges["test_end"],
        "pred_start": pred_dates.min() if not pred_dates.empty else pd.NaT,
        "pred_end": pred_dates.max() if not pred_dates.empty else pd.NaT,
    }


def _build_reg_metrics() -> dict:
    meta = getattr(D, "reg_meta", {}) or {}
    return {
        "model_name": meta.get("model_name", "Regressor"),
        "mae": float(meta.get("mae", 0) or 0),
        "rmse": float(meta.get("rmse", 0) or 0),
        "r2": float(meta.get("r2", 0) or 0),
    }


def _build_model_comparison(m: dict) -> list[tuple[str, float, float]]:
    # If a future notebook export adds comparison results to metadata, use it.
    comparison = (getattr(D, "clf_meta", {}) or {}).get("model_comparison")
    if isinstance(comparison, list) and comparison:
        rows = []
        for item in comparison:
            if isinstance(item, dict):
                rows.append((
                    str(item.get("model", item.get("model_name", "Model"))),
                    float(item.get("pr_auc", 0) or 0),
                    float(item.get("f1", 0) or 0),
                ))
        if rows:
            return rows

    # Minimum dynamic comparison when only selected model metadata is available.
    return [
        ("Random baseline", float(m.get("disruption_rate", 0) or 0), 0.0),
        (str(m.get("model_name", "Selected model")), float(m.get("pr_auc", 0) or 0), float(m.get("f1", 0) or 0)),
    ]


def _build_top_at_risk(limit: int = 5) -> list[tuple[str, str, float]]:
    timetable_df = getattr(D, "timetable_df", pd.DataFrame())
    if timetable_df is None or timetable_df.empty or "disruption_probability" not in timetable_df.columns:
        return []

    df = timetable_df.copy()
    df["disruption_probability"] = pd.to_numeric(df["disruption_probability"], errors="coerce").fillna(0)
    name_col = "station_name" if "station_name" in df.columns else None
    code_col = "station_code" if "station_code" in df.columns else "stanox" if "stanox" in df.columns else None
    if not name_col:
        return []

    idx = df.groupby(name_col)["disruption_probability"].idxmax().dropna().astype(int)
    top = df.loc[idx].sort_values("disruption_probability", ascending=False).head(limit)
    return [
        (str(row[name_col]), str(row[code_col]) if code_col else "", float(row["disruption_probability"]))
        for _, row in top.iterrows()
    ]


CLF_METRICS = _build_clf_metrics()
REG_METRICS = _build_reg_metrics()
MODEL_COMPARISON = _build_model_comparison(CLF_METRICS)
TOP_AT_RISK = _build_top_at_risk()

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _card(label: str, value: str, sub: str = "", colour: str = "#1d70b8") -> str:
    sub_html = f"<div style='font-size:0.7rem;color:#505a5f;margin-top:4px'>{sub}</div>" if sub else ""
    return (
        f"<div style='background:#fff;border:1px solid #dde3ea;border-top:3px solid {colour};"
        f"padding:14px 16px;flex:1 1 160px;min-width:130px'>"
        f"<div style='font-size:0.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;"
        f"color:#505a5f;margin-bottom:6px'>{label}</div>"
        f"<div style='font-size:1.4rem;font-weight:900;color:#1a3a5c;line-height:1'>{value}</div>"
        f"{sub_html}</div>"
    )


def _section(title: str) -> str:
    return (
        f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;"
        f"color:#505a5f;border-bottom:2px solid #1d70b8;padding-bottom:5px;margin:20px 0 12px'>"
        f"{title}</div>"
    )

# ---------------------------------------------------------------------------
# KPI HTML
# ---------------------------------------------------------------------------

def build_kpi_html() -> str:
    m = CLF_METRICS
    r = REG_METRICS

    train_label = _date_label(m["train_start"], m["train_end"])
    test_label = _date_label(m["test_start"], m["test_end"])
    dataset_label = _date_label(m["dataset_start"], m["dataset_end"])
    pred_label = _date_label(m["pred_start"], m["pred_end"])
    threshold_label = f"Mean delay > {m['disruption_threshold_min']:.0f} min"

    # Dataset cards
    dataset_row = (
        "<div style='display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px'>"
        + _card("Station-days (train)", f"{m['train_rows']:,}", train_label, "#1d70b8")
        + _card("Station-days (test)",  f"{m['test_rows']:,}",  test_label, "#f47738")
        + _card("Unique stations",      f"{m['unique_stations']:,}", "Across England", "#1d70b8")
        + _card("Disruption rate",      f"{m['disruption_rate']:.1%}", threshold_label, "#d4351c")
        + _card("Class imbalance",      f"{m['class_imbalance']:.1f}:1", "Not disrupted vs disrupted", "#f47738")
        + "</div>"
    )

    # Classifier cards
    clf_row = (
        "<div style='display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px'>"
        + _card("PR-AUC",    f"{m['pr_auc']:.3f}", "Primary metric (imbalanced data)", "#8b5cf6")
        + _card("ROC-AUC",   f"{m['roc_auc']:.3f}", str(m.get("model_name", "Classifier")), "#8b5cf6")
        + _card("F1 score",  f"{m['f1']:.3f}",      f"Threshold {m['optimal_threshold']:.2f}", "#8b5cf6")
        + "</div>"
    )

    # Regressor cards
    reg_row = (
        "<div style='display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px'>"
        + _card("MAE",  f"{r['mae']:.2f} min", "Mean absolute error (delay)", "#059669")
        + _card("RMSE", f"{r['rmse']:.2f} min","Root mean squared error",     "#059669")
        + _card("R²",   f"{r['r2']:.3f}",      "Variance explained",          "#059669")
        + "</div>"
    )

    note = (
        "<div style='background:#fff8e1;border:1px solid #f59e0b;border-left:4px solid #f59e0b;"
        "padding:10px 14px;font-size:0.78rem;color:#78350f;margin-bottom:10px'>"
        f"<strong>Interpretation:</strong> PR-AUC of {m['pr_auc']:.3f} against a random baseline of "
        f"{m['disruption_rate']:.3f} reflects a difficult imbalanced prediction problem. "
        f"The current class imbalance is {m['class_imbalance']:.1f}:1 across the processed station-day dataset. "
        f"The regression R² is {r['r2']:.3f}, which indicates that open cross-modal data alone explains only a "
        "small share of delay variance without richer operational data such as real-time demand or floating car data."
        "</div>"
    )

    # Prediction window risk bands
    total = m["pred_total"]
    band_bars = ""
    for label, count, colour in [
        ("Low",      m["risk_low"],      "#00703c"),
        ("Moderate", m["risk_moderate"], "#f47738"),
        ("High",     m["risk_high"],     "#d4351c"),
        ("Critical", m["risk_critical"], "#912b11"),
    ]:
        pct = count / total * 100
        band_bars += (
            f"<div style='margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.75rem;"
            f"color:#1a3a5c;margin-bottom:3px'>"
            f"<span style='font-weight:600'>{label}</span><span>{count:,} ({pct:.1f}%)</span></div>"
            f"<div style='background:#eef0f3;height:10px;border-radius:2px'>"
            f"<div style='background:{colour};height:10px;border-radius:2px;width:{pct:.1f}%'></div>"
            f"</div></div>"
        )

    risk_section = (
        f"<div style='background:#fff;border:1px solid #dde3ea;padding:14px 16px;margin-bottom:10px'>"
        f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;"
        f"color:#505a5f;margin-bottom:10px'>Prediction window risk bands "
        f"<span style='font-weight:400'>({total:,} station-days, {pred_label})</span></div>"
        f"{band_bars}</div>"
    )

    # Top 5 at-risk stations
    top5_rows = ""
    for i, (name, code, prob) in enumerate(TOP_AT_RISK):
        bg    = "#fff" if i % 2 == 0 else "#f0f4f8"
        bar_w = int(prob * 100)
        top5_rows += (
            f"<tr style='background:{bg}'>"
            f"<td style='padding:7px 10px;font-size:0.78rem;color:#1a3a5c;font-weight:600'>{name}</td>"
            f"<td style='padding:7px 10px;font-size:0.78rem;color:#505a5f'>{code}</td>"
            f"<td style='padding:7px 10px'>"
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<div style='background:#eef0f3;height:8px;width:80px;border-radius:2px'>"
            f"<div style='background:#d4351c;height:8px;width:{bar_w}px;border-radius:2px'></div></div>"
            f"<span style='font-size:0.78rem;font-weight:700;color:#d4351c'>{prob:.1%}</span>"
            f"</div></td></tr>"
        )

    top5 = (
        "<div style='background:#fff;border:1px solid #dde3ea;margin-bottom:10px'>"
        "<div style='background:#1a3a5c;padding:8px 12px;font-size:0.72rem;font-weight:700;"
        "letter-spacing:.06em;text-transform:uppercase;color:#fff'>Top 5 highest-risk stations "
        "<span style='font-weight:400;opacity:.7'>(peak disruption probability)</span></div>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<thead><tr style='background:#f4f7fb'>"
        "<th style='padding:7px 10px;text-align:left;font-size:0.72rem;color:#505a5f'>Station</th>"
        "<th style='padding:7px 10px;text-align:left;font-size:0.72rem;color:#505a5f'>CRS</th>"
        "<th style='padding:7px 10px;text-align:left;font-size:0.72rem;color:#505a5f'>Peak probability</th>"
        f"</tr></thead><tbody>{top5_rows}</tbody></table></div>"
    )

    return (
        _section(f"Dataset summary ({dataset_label})")
        + dataset_row
        + _section("XGBoost classifier performance")
        + clf_row
        + _section("XGBoost regressor performance")
        + reg_row
        + note
        + _section(f"Prediction window risk breakdown ({pred_label})")
        + risk_section
        + top5
    )

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def build_split_fig() -> plt.Figure:
    m = CLF_METRICS
    fig, ax = plt.subplots(figsize=(9, 1.8))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    rows = [
        (f"Training   {_date_label(m['train_start'], m['train_end'])}   {m['train_rows']:,} rows",
         m["train_start"], m["train_end"], "#1d70b8", 0.55),
        (f"Test   {_date_label(m['test_start'], m['test_end'])}   {m['test_rows']:,} rows",
         m["test_start"], m["test_end"], "#f47738", 0.55),
        (f"Prediction   {_date_label(m['pred_start'], m['pred_end'])}   {m['pred_total']:,} rows",
         m["pred_start"], m["pred_end"], "#8b5cf6", 0.25),
    ]

    valid_dates = []
    for label, start, end, colour, y in rows:
        if pd.isna(start) or pd.isna(end):
            continue
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        valid_dates.extend([s, e])
        width_days = max((e - s).days, 1)
        ax.barh(y, width_days, left=s, height=0.18, color=colour, alpha=0.85, edgecolor="white")
        ax.text(s + pd.Timedelta(days=width_days / 2), y + 0.11, label, ha="center", va="bottom",
                fontsize=6.5, color=colour, fontweight="bold")

    if valid_dates:
        ax.set_xlim(min(valid_dates) - pd.Timedelta(days=2), max(valid_dates) + pd.Timedelta(days=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.set_ylim(0.1, 0.85)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.set_title("Train / test / prediction split", fontsize=8.5, fontweight="bold", pad=6)
    plt.tight_layout()
    return fig


def build_imbalance_fig() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    counts = [CLF_METRICS["not_disrupted"], CLF_METRICS["disrupted"]]
    bars = ax.bar(
        ["Not disrupted", "Disrupted"], counts,
        color=["#1d70b8", "#d4351c"], alpha=0.82, edgecolor="white", width=0.5,
    )
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{cnt:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Station-days", fontsize=8)
    ax.set_title("Class distribution (train + test)", fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, max(CLF_METRICS["not_disrupted"], 1) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.95, f"Imbalance ratio: {CLF_METRICS['class_imbalance']:.0f}:1",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#d4351c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0", edgecolor="#d4351c", alpha=0.8))
    plt.tight_layout()
    return fig


def build_comparison_fig() -> plt.Figure:
    models  = [r[0] for r in MODEL_COMPARISON]
    pr_vals = [r[1] for r in MODEL_COMPARISON]
    f1_vals = [r[2] for r in MODEL_COMPARISON]
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    b1 = ax.bar(x - w / 2, pr_vals, w, label="PR-AUC",                color="#8b5cf6", alpha=0.82, edgecolor="white")
    b2 = ax.bar(x + w / 2, f1_vals, w, label="F1 (optimal threshold)", color="#1d70b8", alpha=0.82, edgecolor="white")
    for bar in list(b1) + list(b2):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5)

    selected_model = str(CLF_METRICS.get("model_name", ""))
    selected_idx = next((i for i, name in enumerate(models) if name == selected_model), len(models) - 1)
    if selected_idx >= 0:
        ax.axvspan(selected_idx - 0.5, selected_idx + 0.5, alpha=0.06, color="#8b5cf6")
        y_label = max(pr_vals + f1_vals + [0.05]) * 1.08
        ax.text(selected_idx, y_label, "Selected", ha="center", fontsize=7, color="#8b5cf6", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7.5)
    ax.set_ylabel("Score", fontsize=8)
    y_max = max(pr_vals + f1_vals + [CLF_METRICS["disruption_rate"], 0.05]) * 1.35
    ax.set_ylim(0, y_max)
    ax.set_title("Model comparison - PR-AUC and F1", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(CLF_METRICS["disruption_rate"], color="grey", linestyle="--", lw=1, alpha=0.7)
    ax.text(0.01, CLF_METRICS["disruption_rate"] + 0.003, "Random baseline",
            transform=ax.get_yaxis_transform(), fontsize=6.5, color="grey")
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Gradio view
# ---------------------------------------------------------------------------

def build_data_view():
    split_fig   = build_split_fig()
    imbal_fig   = build_imbalance_fig()
    compare_fig = build_comparison_fig()

    with gr.Group(visible=False) as v2:
        with gr.Column(elem_classes=["rr-content"]):
            gr.HTML('<div class="rr-panel-title">Dataset and model performance dashboard</div>')
            gr.HTML(build_kpi_html())

            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    gr.HTML('<div style="font-size:0.68rem;font-weight:700;letter-spacing:.1em;'
                            'text-transform:uppercase;color:#505a5f;border-bottom:2px solid #1d70b8;'
                            'padding-bottom:5px;margin:0 0 10px">Data split timeline</div>')
                    gr.Plot(value=split_fig, label="")
                with gr.Column(scale=3):
                    gr.HTML('<div style="font-size:0.68rem;font-weight:700;letter-spacing:.1em;'
                            'text-transform:uppercase;color:#505a5f;border-bottom:2px solid #1d70b8;'
                            'padding-bottom:5px;margin:0 0 10px">Class distribution</div>')
                    gr.Plot(value=imbal_fig, label="")

            gr.HTML('<div style="font-size:0.68rem;font-weight:700;letter-spacing:.1em;'
                    'text-transform:uppercase;color:#505a5f;border-bottom:2px solid #1d70b8;'
                    'padding-bottom:5px;margin:16px 0 10px">Model comparison</div>')
            gr.Plot(value=compare_fig, label="")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML('<div style="font-size:0.68rem;font-weight:700;letter-spacing:.1em;'
                            'text-transform:uppercase;color:#505a5f;border-bottom:2px solid #1d70b8;'
                            'padding-bottom:5px;margin:16px 0 10px">Classifier feature importance (XGBoost)</div>')
                    gr.Plot(value=CLF_FIG, label="")
                with gr.Column(scale=1):
                    gr.HTML('<div style="font-size:0.68rem;font-weight:700;letter-spacing:.1em;'
                            'text-transform:uppercase;color:#505a5f;border-bottom:2px solid #1d70b8;'
                            'padding-bottom:5px;margin:16px 0 10px">Regressor feature importance (XGBoost tuned)</div>')
                    gr.Plot(value=REG_FIG, label="")


    return v2