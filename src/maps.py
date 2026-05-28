import pandas as pd

try:
    import folium
    from folium.plugins import MiniMap, Fullscreen
    _OK = True
except ImportError:
    _OK = False

_UK        = [54.5, -2.5]
_UK_BOUNDS = [[49.5, -8.5], [61.2, 2.5]]
_H         = "600px"

NO_FOLIUM = "<p style='color:#d4351c;padding:20px;font-family:Arial'>Install folium to see maps.</p>"

PLACEHOLDER = """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            height:600px;background:#f3f2f1;border:2px dashed #b1b4b6;color:#505a5f;
            font-family:Arial,sans-serif;gap:10px;">
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#1d70b8" stroke-width="1.5">
    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/>
    <circle cx="12" cy="9" r="2.5"/>
  </svg>
  <span style="font-size:0.95rem;font-weight:600;color:#0b0c0c">Select filters and click Load</span>
  <span style="font-size:0.8rem;color:#505a5f">The network map will appear here</span>
</div>"""

_C_UNPLANNED = "#d4351c"
_C_PLANNED   = "#f47738"
_C_STATION   = "#1d70b8"
_C_LINK      = "#505a5f"
_C_RADIUS_10 = "#1d70b8"
_C_RADIUS_25 = "#6b7280"

_RISK_COLOUR = {
    "low":      "#00703c",
    "moderate": "#f47738",
    "high":     "#d4351c",
    "critical": "#912b11",
}


def _risk_colour(risk: str) -> str:
    return _RISK_COLOUR.get(str(risk).lower(), _C_STATION)


def _legend(items):
    rows = []
    for item in items:
        label, colour = item[0], item[1]
        shape = item[2] if len(item) > 2 else "circle"
        if shape == "diamond":
            swatch = (
                f"<span style='width:12px;height:12px;background:{colour};"
                "display:inline-block;flex-shrink:0;transform:rotate(45deg);"
                "border:1px solid rgba(0,0,0,0.3)'></span>"
            )
        elif shape == "square":
            swatch = (
                f"<span style='width:12px;height:12px;background:{colour};"
                "display:inline-block;flex-shrink:0;border:1px solid rgba(0,0,0,0.3)'></span>"
            )
        elif shape == "line":
            swatch = (
                f"<span style='width:18px;height:0;border-top:3px solid {colour};"
                "display:inline-block;flex-shrink:0'></span>"
            )
        elif shape == "dash":
            swatch = (
                f"<span style='width:18px;height:0;border-top:2px dashed {colour};"
                "display:inline-block;flex-shrink:0'></span>"
            )
        elif shape == "ring":
            swatch = (
                f"<span style='width:12px;height:12px;border-radius:50%;background:transparent;"
                f"border:2px solid {colour};display:inline-block;flex-shrink:0'></span>"
            )
        else:
            swatch = (
                f"<span style='width:12px;height:12px;border-radius:50%;background:{colour};"
                "display:inline-block;flex-shrink:0;border:1px solid rgba(0,0,0,0.3)'></span>"
            )
        rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
            f"{swatch}<span style='font-size:12px;color:#0b0c0c'>{label}</span></div>"
        )
    return (
        "<div style='position:absolute;bottom:30px;left:10px;z-index:999;"
        "background:#fff;padding:10px 14px;border:1px solid #b1b4b6;"
        "border-left:4px solid #1d70b8;font-family:Arial,sans-serif;"
        f"box-shadow:0 2px 6px rgba(0,0,0,.15)'>{''.join(rows)}</div>"
    )


def _plugins(m):
    MiniMap(tile_layer="CartoDB positron", toggle_display=True, minimized=True).add_to(m)
    Fullscreen(position="topright").add_to(m)


def _wrap(m):
    return (f"<div style='width:100%;height:{_H};border:1px solid #b1b4b6;"
            f"border-top:3px solid #1d70b8;overflow:hidden'>{m._repr_html_()}</div>")


def _map(loc, zoom):
    m = folium.Map(location=loc, zoom_start=zoom, tiles="CartoDB positron",
                   prefer_canvas=True, control_scale=True, min_zoom=5, max_zoom=16,
                   max_bounds=True)
    m.options["maxBounds"] = _UK_BOUNDS
    return m


def _closure_popup(cl):
    ctype = cl.get("closure_type", "")
    bg    = _C_UNPLANNED if ctype == "unplanned" else _C_PLANNED
    st    = cl.get("start_time_dt", "")
    start = st.strftime("%d %b %Y %H:%M UTC") if hasattr(st, "strftime") else str(st)[:16]
    return (
        f"<div style='font-family:Arial;min-width:210px'>"
        f"<div style='background:#0b0c0c;color:#fff;padding:8px 10px;margin:-10px -10px 8px;"
        f"font-weight:700'>{cl.get('road_name','')}"
        f"<span style='background:{bg};font-size:10px;padding:2px 6px;margin-left:6px;"
        f"text-transform:uppercase'>{ctype}</span></div>"
        f"<table style='font-size:12px;border-collapse:collapse;width:100%'>"
        f"<tr><td style='color:#505a5f;padding:3px 4px'>Cause</td>"
        f"    <td style='font-weight:700;padding:3px 4px'>{cl.get('cause_type','')}</td></tr>"
        f"<tr><td style='color:#505a5f;padding:3px 4px'>Duration</td>"
        f"    <td style='font-weight:700;padding:3px 4px'>{float(cl.get('duration_hours',0)):.1f} h</td></tr>"
        f"<tr><td style='color:#505a5f;padding:3px 4px'>Start</td>"
        f"    <td style='font-weight:700;padding:3px 4px'>{start}</td></tr>"
        f"</table></div>"
    )


def _closure_marker(location, cl, colour, size=14, selected=False):
    """Diamond marker using DivIcon - works in all folium versions."""
    border_col = "#0b0c0c" if selected else "rgba(0,0,0,0.4)"
    border_w   = "2.5px" if selected else "1.5px"
    icon_html  = (
        f"<div style='width:{size}px;height:{size}px;background:{colour};"
        f"transform:rotate(45deg);border:{border_w} solid {border_col};"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.4)'></div>"
    )
    icon = folium.DivIcon(
        html=icon_html,
        icon_size=(size, size),
        icon_anchor=(size // 2, size // 2),
    )
    marker = folium.Marker(
        location=location,
        icon=icon,
        popup=folium.Popup(_closure_popup(cl), max_width=280),
        tooltip=folium.Tooltip(
            f"<b>{cl.get('road_name','')}</b> - {cl.get('closure_type','')}<br>"
            f"<span style='font-size:11px'>{cl.get('cause_type','')}</span>",
            sticky=True,
        ),
    )
    return marker


def _station_marker(location, station_name, risk="N/A", dist=None, colour=None, radius=8):
    """Circle marker for rail stations via CircleMarker."""
    c = colour or _risk_colour(risk)
    dist_txt = f"<br>{dist} km from closure" if dist not in (None, "?") else ""
    risk_txt = str(risk).upper()
    return folium.CircleMarker(
        location,
        radius=radius,
        color="#0b0c0c",
        fill=True,
        fill_color=c,
        fill_opacity=0.88,
        weight=1.4,
        popup=folium.Popup(
            f"<b>{station_name}</b><br>Risk: <b>{risk_txt}</b>{dist_txt}",
            max_width=220,
        ),
        tooltip=folium.Tooltip(
            f"<b>{station_name}</b><br>{dist_txt.replace('<br>', '')} · {risk_txt}" if dist_txt else f"<b>{station_name}</b>",
            sticky=True,
        ),
    )


def _add_radius_rings(m, lat, lon, show_10=True, show_25=True):
    """Draw 10 km and 25 km closure impact rings."""
    rg = folium.FeatureGroup(name="10 km / 25 km closure radius", show=True)
    if show_25:
        folium.Circle(
            [lat, lon], radius=25_000,
            color=_C_RADIUS_25, weight=2, opacity=0.75,
            fill=True, fill_color=_C_RADIUS_25, fill_opacity=0.035,
            tooltip="25 km radius from closure",
        ).add_to(rg)
    if show_10:
        folium.Circle(
            [lat, lon], radius=10_000,
            color=_C_RADIUS_10, weight=2, opacity=0.95,
            fill=True, fill_color=_C_RADIUS_10, fill_opacity=0.055,
            tooltip="10 km radius from closure",
        ).add_to(rg)
    rg.add_to(m)
    return rg


def overview(active: pd.DataFrame) -> str:
    if not _OK: return NO_FOLIUM
    if active.empty: return PLACEHOLDER
    m = _map(_UK, 6)
    m.fit_bounds(_UK_BOUNDS)
    _plugins(m)
    ug = folium.FeatureGroup(name="Unplanned", show=True)
    pg = folium.FeatureGroup(name="Planned",   show=True)
    for _, cl in active.iterrows():
        unp = cl["closure_type"] == "unplanned"
        c   = _C_UNPLANNED if unp else _C_PLANNED
        sz  = 14 if unp else 11
        _closure_marker(
            [cl["closure_lat"], cl["closure_lon"]], cl, c, size=sz
        ).add_to(ug if unp else pg)
    ug.add_to(m); pg.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.get_root().html.add_child(folium.Element(_legend([
        ("Unplanned closure", _C_UNPLANNED, "diamond"),
        ("Planned closure",   _C_PLANNED,   "diamond"),
    ])))
    return _wrap(m)


def station_detail(st_lat, st_lon, station_name, nearby_closures: pd.DataFrame) -> str:
    if not _OK: return NO_FOLIUM
    m = _map([st_lat, st_lon], 9)
    _plugins(m)
    # Station marker: outer ring + filled dot
    folium.CircleMarker([st_lat, st_lon], radius=18, color=_C_STATION,
                        fill=False, weight=2, opacity=0.3).add_to(m)
    folium.CircleMarker([st_lat, st_lon], radius=10, color=_C_STATION,
                        fill=True, fill_color=_C_STATION, fill_opacity=0.92, weight=2,
                        popup=folium.Popup(f"<b>{station_name}</b>", max_width=200),
                        tooltip=folium.Tooltip(f"<b>{station_name}</b>", sticky=True)).add_to(m)
    ug = folium.FeatureGroup(name="Unplanned closures", show=True)
    pg = folium.FeatureGroup(name="Planned closures",   show=True)
    lg = folium.FeatureGroup(name="Proximity links",    show=True)
    for _, cl in nearby_closures.iterrows():
        if pd.isna(cl.get("closure_lat")) or pd.isna(cl.get("closure_lon")): continue
        unp = cl.get("closure_type", "") == "unplanned"
        c   = _C_UNPLANNED if unp else _C_PLANNED
        folium.PolyLine([[st_lat, st_lon], [cl["closure_lat"], cl["closure_lon"]]],
                        color=_C_LINK, weight=1.5, opacity=0.4, dash_array="6 4").add_to(lg)
        _add_radius_rings(m, float(cl["closure_lat"]), float(cl["closure_lon"]), show_10=True, show_25=True)
        _closure_marker(
            [cl["closure_lat"], cl["closure_lon"]], cl, c, size=12
        ).add_to(ug if unp else pg)
    ug.add_to(m); pg.add_to(m); lg.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.get_root().html.add_child(folium.Element(
        _legend([
            ("Selected station",    _C_STATION,   "circle"),
            ("Unplanned closure",   _C_UNPLANNED, "diamond"),
            ("Planned closure",     _C_PLANNED,   "diamond"),
            ("10 km closure radius",_C_RADIUS_10, "ring"),
            ("25 km closure radius",_C_RADIUS_25, "ring"),
        ])))
    return _wrap(m)


def closure_detail(cl_row, nearby_df: pd.DataFrame, stations_ref: pd.DataFrame,
                   risk_by_station: dict = None) -> str:
    """Closure-centric map with risk-coloured station circles."""
    if not _OK: return NO_FOLIUM
    lat  = float(cl_row["closure_lat"])
    lon  = float(cl_row["closure_lon"])
    m    = _map([lat, lon], 9)
    _plugins(m)

    _add_radius_rings(m, lat, lon, show_10=True, show_25=True)
    closure_colour = _C_UNPLANNED if cl_row.get("closure_type", "") == "unplanned" else _C_PLANNED
    _closure_marker([lat, lon], cl_row, closure_colour, size=18, selected=True).add_to(m)

    sg = folium.FeatureGroup(name="Nearby stations", show=True)
    lg = folium.FeatureGroup(name="Proximity links", show=True)

    if not nearby_df.empty:
        for _, nb in nearby_df.iterrows():
            sr = stations_ref[stations_ref["station_name"] == nb["station_name"]]
            if sr.empty: continue
            sl   = float(sr.iloc[0]["latitude"])
            sn   = float(sr.iloc[0]["longitude"])
            stn  = nb["station_name"]
            risk = (risk_by_station or {}).get(stn, "N/A")
            c    = _risk_colour(risk) if risk_by_station else _C_STATION
            dist = nb.get("distance_km", "?")

            folium.PolyLine([[lat, lon], [sl, sn]], color=_C_LINK, weight=1.5,
                            opacity=0.45, dash_array="6 4",
                            tooltip=f"{dist} km").add_to(lg)
            _station_marker([sl, sn], stn, risk=risk, dist=dist, colour=c, radius=8).add_to(sg)

    sg.add_to(m); lg.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    selected_colour = _C_UNPLANNED if cl_row.get("closure_type", "") == "unplanned" else _C_PLANNED
    legend_items = [
        ("Selected road closure", selected_colour, "diamond"),
        ("10 km radius",          _C_RADIUS_10,   "ring"),
        ("25 km radius",          _C_RADIUS_25,   "ring"),
        ("Distance link",         _C_LINK,        "dash"),
    ]
    if risk_by_station:
        legend_items += [
            ("Low risk station",      _RISK_COLOUR["low"],      "circle"),
            ("Moderate risk station", _RISK_COLOUR["moderate"], "circle"),
            ("High risk station",     _RISK_COLOUR["high"],     "circle"),
            ("Critical risk station", _RISK_COLOUR["critical"], "circle"),
        ]
    else:
        legend_items.append(("Nearby station", _C_STATION, "circle"))

    m.get_root().html.add_child(folium.Element(_legend(legend_items)))
    return _wrap(m)