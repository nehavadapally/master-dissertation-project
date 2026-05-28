import html

import gradio as gr
import pandas as pd

import src.data as D
import src.maps as M
from src.llm_summary import generate_briefing

# ---------------------------------------------------------------------------
# Static HTML constants
# ---------------------------------------------------------------------------

TOPBAR = """
<div class="rr-topbar">
  <div class="rr-logo">RR</div>
  <div>
    <h1>Road Rail Resilience Service</h1>
    <p>Cross-modal disruption early warning &nbsp;| April 2026</p>
  </div>
</div>
<div class="rr-pills">
  <span>Data: 3 Apr - 30 Apr 2026</span>
  <span>Radius: 10-25 km</span>
  <span>Model: XGBoost classifier + regression</span>
  <span>Sources: DATEX II · TRUST · Darwin</span>
</div>"""

FOOTER = (
    "<div style='border-top:1px solid #b1b4b6;padding:7px 20px;font-size:0.68rem;"
    "color:#505a5f;background:#fff'>"
    "National Highways DATEX II &middot; Network Rail TRUST &middot; "
    "Darwin CIF &middot; April 2026"
    "</div>"
)

GAP_FIX_JS = """<script>
(function(){
  function fix(){
    var m=document.getElementById('rr-main');
    if(m){ m.style.gap='0px'; m.style.rowGap='0px'; m.style.padding='0px'; }
    document.querySelectorAll(
      '#rr-main > .block > .hide, #rr-main > .block > .hidden,'+
      '#rr-main > div > .block > .hide, #rr-main > div > .block > .hidden'
    ).forEach(function(el){
      var b=el.parentElement;
      if(b&&(b.classList.contains('block')||b.classList.contains('form'))){
        b.style.cssText+=';display:none!important;height:0!important;'+
          'min-height:0!important;max-height:0!important;'+
          'padding:0!important;margin:0!important;border:none!important;overflow:hidden!important';
      }
    });
    document.querySelectorAll('#rr-main > .block, #rr-main > div > .block').forEach(function(b){
      if(!b.querySelector(':scope > .hide')&&!b.querySelector(':scope > .hidden')){
        b.style.removeProperty('display');
        b.style.removeProperty('height');
        b.style.removeProperty('max-height');
      }
    });
  }
  var obs=new MutationObserver(fix);
  function init(){
    fix();
    obs.observe(document.getElementById('rr-main')||document.body,
      {subtree:true,attributes:true,attributeFilter:['class','style']});
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init);
  else init();
  setTimeout(fix,400); setTimeout(fix,1000);
})();
</script>"""

# placeholder HTML strings
_NOTE_EMPTY  = "<div class='rr-note'>Select filters and click Load.</div>"
_NOTE_NO_STN = "<div class='rr-note'>Click a closure row to see nearby stations.</div>"
_PRED_EMPTY  = ("<div class='rr-pred'><p style='color:#505a5f;font-size:0.83rem;margin:0'>"
                "Select a station from the table below.</p></div>")


def _note(html: str) -> str:
    return f"<div class='rr-note'>{html}</div>"


def _station_context(station: str, planned_date) -> str:
    """Small reusable header shown inside station-specific panels."""
    stn = html.escape(str(station or "Selected station"))
    try:
        date_label = planned_date.strftime("%A %d %B %Y")
    except Exception:
        date_label = html.escape(str(planned_date or ""))
    return (
        "<div style='background:#f4f7fb;border:1px solid #dde3ea;"
        "border-left:4px solid #1d70b8;padding:8px 12px;margin-bottom:10px;"
        "font-family:Arial,sans-serif'>"
        "<div style='font-size:0.62rem;font-weight:700;letter-spacing:.12em;"
        "text-transform:uppercase;color:#505a5f;margin-bottom:4px'>Selected station</div>"
        f"<div style='font-size:0.92rem;font-weight:800;color:#1a3a5c'>{stn}</div>"
        f"<div style='font-size:0.72rem;color:#505a5f;margin-top:2px'>{date_label}</div>"
        "</div>"
    )


def _prediction_panel(station: str, planned_date, badge_html: str, briefing_html: str = "") -> str:
    """Prediction panel with optional operational briefing shown in the same section."""
    briefing_block = ""
    if briefing_html:
        briefing_block = (
            "<div style='margin-top:12px;border-top:1px solid #dde3ea;padding-top:10px'>"
            f"{briefing_html}"
            "</div>"
        )

    return (
        f'<div class="rr-pred">'
        f'{_station_context(station, planned_date)}'
        f'{badge_html}'
        f'{briefing_block}'
        f'</div>'
    )


def _briefing_panel(station: str, planned_date, briefing_html: str) -> str:
    return f"{_station_context(station, planned_date)}{briefing_html}"

# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def on_view_switch(sel: str):
    map_view = sel == "Road Closures & Stations"
    return gr.update(visible=map_view), gr.update(visible=not map_view)


def on_load(date_str, time_str, ctype, max_dist, max_dur, stn):
    dt  = D.combine_datetime(date_str, time_str)
    d   = D.parse_date(dt)
    stn_active = isinstance(stn, str) and stn.strip()

    if stn_active:
        cl_tbl, cl_note, raw, _ = D.get_closures_near_station(stn, dt, ctype, max_dist, max_dur)
        map_html = M.station_detail(
            *D.stations_ref[D.stations_ref["station_name"] == stn][["latitude", "longitude"]].iloc[0],
            stn,
            pd.DataFrame(raw) if raw else pd.DataFrame(),
        )
        badge, wa, wf, wb, tf = D.get_station_panels(stn, d)
        risk      = D.get_station_risk(stn, d)
        closures  = raw
        mean_h    = D.station_day_df[D.station_day_df["station_name"] == stn]["mean_delay_minutes"].mean() if not D.station_day_df[D.station_day_df["station_name"] == stn].empty else None
        late_h    = D.station_day_df[D.station_day_df["station_name"] == stn]["late_share"].mean() if mean_h is not None else None
        llm_html  = generate_briefing(stn, d, risk["risk"], risk["prob"], risk["disrupted"], risk["delay"], closures, mean_h, late_h)
        return (
            cl_tbl, _note(cl_note), raw, map_html,
            pd.DataFrame(), _note(f"Predictions for <b>{stn}</b>"), gr.update(visible=False),
            _prediction_panel(stn, d, badge, llm_html),
            wa, wf, wb, tf,
        )

    cl_tbl, cl_note, raw, _ = D.get_closures(dt, ctype, max_dist, max_dur)
    map_html = M.overview(pd.DataFrame(raw) if raw else pd.DataFrame())
    return (
        cl_tbl, _note(cl_note), raw, map_html,
        pd.DataFrame(), _NOTE_NO_STN, gr.update(visible=True),
        _PRED_EMPTY,
        pd.DataFrame(), None, pd.DataFrame(), None,
    )


def on_closure_row_select(disp_df, active_records, max_dist, stn_filter, date_str, time_str, max_dur, evt: gr.SelectData):
    dt = D.combine_datetime(date_str, time_str)
    d  = D.parse_date(dt)

    if isinstance(stn_filter, str) and stn_filter.strip():
        return pd.DataFrame(), _NOTE_NO_STN, M.PLACEHOLDER

    try:
        row_idx = int(evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index)
    except Exception as e:
        return pd.DataFrame(), _note(f"Row error: {e}"), M.PLACEHOLDER

    if not active_records or disp_df is None or len(disp_df) == 0:
        return pd.DataFrame(), _note("Click Load first."), M.PLACEHOLDER

    active = pd.DataFrame(active_records)
    for col in ["closure_lat", "closure_lon", "duration_hours"]:
        if col in active.columns:
            active[col] = pd.to_numeric(active[col], errors="coerce")
    active["duration_hours"] = active["duration_hours"].fillna(0)
    if "start_time_dt" in active.columns:
        active["start_time_dt"] = pd.to_datetime(active["start_time_dt"], utc=True, errors="coerce")
    active["situation_id"] = active["situation_id"].astype(str)

    try:
        sid  = str(disp_df.iloc[row_idx]["ID"])
        road = str(disp_df.iloc[row_idx]["Road"])
    except Exception as e:
        return pd.DataFrame(), _note(f"Row error: {e}"), M.PLACEHOLDER

    cl = active[active["situation_id"] == sid]
    if cl.empty:
        return pd.DataFrame(), _note(f"ID {sid} not found."), M.PLACEHOLDER

    clat   = cl.iloc[0].get("closure_lat")
    clon   = cl.iloc[0].get("closure_lon")
    radius = float(max_dist) if max_dist is not None else 25.0

    if pd.isna(clat) or pd.isna(clon):
        return pd.DataFrame(), _note(f"{road}: no coordinates."), M.PLACEHOLDER

    nb = D.get_nearby_stations(cl, max_km=radius)
    if nb.empty:
        return pd.DataFrame(), _note("Station reference empty."), M.PLACEHOLDER

    if len(nb) == 0:
        return pd.DataFrame(), _note(f"No stations within {radius:.0f} km of {road}."), M.PLACEHOLDER

    # build risk lookup for the selected date
    risk_by_station = {
        row["station_name"]: row.get("risk_band", "N/A")
        for _, row in D.timetable_df[D.timetable_df["planned_date"] == d].iterrows()
        if row["station_name"] in set(nb["station_name"])
    }

    disp     = nb[["station_name", "distance_km", "road", "type"]].copy()
    disp.columns = ["Station", "Distance (km)", "Road", "Type"]
    map_html = M.closure_detail(cl.iloc[0], nb, D.stations_ref, risk_by_station)

    return (
        disp.reset_index(drop=True),
        _note(f"<b>{len(disp)}</b> station(s) within {radius:.0f} km of {road} - click a row for predictions"),
        map_html,
    )


def on_station_row_select(nb_df, date_str, time_str, evt: gr.SelectData):
    _empty = _PRED_EMPTY, pd.DataFrame(), None, pd.DataFrame(), None

    try:
        row_idx = int(evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index)
    except Exception:
        return _empty

    if nb_df is None or len(nb_df) == 0:
        return _empty

    try:
        stn = str(nb_df.iloc[row_idx]["Station"])
    except Exception:
        return _empty

    dt   = D.combine_datetime(date_str, time_str)
    d    = D.parse_date(dt)
    risk = D.get_station_risk(stn, d)

    badge, wa, wf, wb, tf = D.get_station_panels(stn, d)

    _, _, nearby_closures, _ = D.get_closures_near_station(stn, dt[:10], "All", 25, 72)
    hist = D.station_day_df[D.station_day_df["station_name"] == stn]
    mean_h = float(hist["mean_delay_minutes"].mean()) if not hist.empty else None
    late_h = float(hist["late_share"].mean())         if not hist.empty else None

    llm_html = generate_briefing(
        stn, d, risk["risk"], risk["prob"], risk["disrupted"], risk["delay"],
        nearby_closures, mean_h, late_h,
    )
    return _prediction_panel(stn, d, badge, llm_html), wa, wf, wb, tf

# ---------------------------------------------------------------------------
# Layout builders
# ---------------------------------------------------------------------------

def build_sidebar():
    with gr.Column(scale=0, min_width=155, elem_classes=["rr-side"], elem_id="rr-side"):
        gr.HTML('<div class="rr-nav-label">View</div>')
        view_sel = gr.Radio(
            choices=["Road Closures & Stations", "Data Overview"],
            value="Road Closures & Stations",
            label="", container=False, elem_classes=["rr-radio"],
        )
    return view_sel


def build_filterbar():
    with gr.Group(visible=True, elem_id="rr-filterbar") as filterbar:
        with gr.Row(equal_height=True, elem_id="rr-filterbar-row"):
            with gr.Column(min_width=155, elem_classes=["rr-fb-col"]):
                f_date = gr.Textbox(value="2026-04-11", label="Date (UTC)", interactive=True,
                                    placeholder="YYYY-MM-DD", elem_classes=["rr-date-picker"],
                                    html_attributes={"type": "date", "min": "2026-04-03", "max": "2026-04-30"})
            with gr.Column(min_width=110, elem_classes=["rr-fb-col"]):
                f_time = gr.Textbox(value="", label="Time (UTC)", interactive=True,
                                    placeholder="HH:MM", elem_classes=["rr-time-picker"],
                                    html_attributes={"type": "time"})
            with gr.Column(min_width=140, elem_classes=["rr-fb-col"]):
                f_type = gr.Dropdown(choices=["All", "Planned", "Unplanned"], value="All", label="Closure type")
            with gr.Column(min_width=120, elem_classes=["rr-fb-col"]):
                f_dist = gr.Number(value=25, label="Radius (km)", minimum=10, maximum=25, step=1, precision=0, elem_classes=["rr-fb-number"])
            with gr.Column(min_width=120, elem_classes=["rr-fb-col"]):
                f_dur = gr.Number(value=72, label="Max duration (hrs)", minimum=0, maximum=72, step=1, precision=0, elem_classes=["rr-fb-number"])
            with gr.Column(min_width=200, elem_classes=["rr-fb-col", "rr-fb-col-last"]):
                f_stn = gr.Dropdown(choices=D.ALL_STATIONS, value=None, label="Station",
                                    filterable=True, allow_custom_value=False)
        with gr.Row(elem_id="rr-filterbar-btnrow"):
            gr.HTML("<span style='font-size:0.68rem;color:#505a5f;align-self:center'>"
                    "Time filters closures active at that moment</span>")
            go_btn = gr.Button("Load", variant="primary", elem_id="rr-load-btn")
    return filterbar, f_date, f_time, f_type, f_dist, f_dur, f_stn, go_btn


def build_map_view():
    with gr.Group(visible=True) as v1:
        _map1 = gr.HTML(value=M.PLACEHOLDER, elem_classes=["rr-map-wrap"])

        with gr.Row(elem_classes=["rr-content"], equal_height=False):
            with gr.Column(scale=6, min_width=300):
                gr.HTML('<div class="rr-panel-title">Closures - click a row</div>')
                _note1 = gr.HTML(_NOTE_EMPTY)
                _raw1  = gr.State([])
                _tbl1  = gr.Dataframe(interactive=False, wrap=True, max_height=260, label="")
            with gr.Column(scale=4, min_width=240, visible=True) as _nb_col:
                gr.HTML('<div class="rr-panel-title">Nearby stations</div>')
                _nb_note = gr.HTML(_NOTE_NO_STN)
                gr.HTML('<div class="rr-clickable-hint">Click a row to load predictions</div>')
                _nb_tbl  = gr.Dataframe(interactive=False, wrap=True, max_height=200, label="")

        with gr.Accordion("Disruption prediction", open=True):
            _pred1 = gr.HTML(_PRED_EMPTY)

        with gr.Accordion("Historical performance", open=False):
            _hfig1 = gr.Plot(label="")
            _htbl1 = gr.Dataframe(interactive=False, wrap=False, max_height=200, label="")

        with gr.Accordion("Timetable predictions", open=False):
            _tfig1 = gr.Plot(label="")
            _ttbl1 = gr.Dataframe(interactive=False, wrap=False, max_height=200, label="")

    return v1, _map1, _note1, _raw1, _tbl1, _nb_col, _nb_note, _nb_tbl, _pred1, _htbl1, _hfig1, _ttbl1, _tfig1