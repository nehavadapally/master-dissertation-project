"""Geospatial utilities - haversine distance and closure-to-station matching."""

import numpy as np
import pandas as pd


def haversine_vectorised(
    lat1: float, lon1: float,
    lat2_array: np.ndarray, lon2_array: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine distance (km) from one point to an array of points."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2 = np.radians(lat2_array)
    lon2 = np.radians(lon2_array)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def find_nearby_stations(
    road_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    min_km: float = 10,
    max_km: float = 25,
) -> pd.DataFrame:
    """Expand road closures by matching each to stations within a distance band.

    Args:
        road_df:      Road closure DataFrame. Required columns: closure_lat,
                      closure_lon, situation_id, closure_type, start_time,
                      end_time, road_name, road_class, closure_severity,
                      start_hour, start_dow, start_date, duration_hours,
                      effective_duration_hours, effective_end_time,
                      effective_start_time, validity_status, cause_type.
        stations_df:  Station reference DataFrame. Required columns: latitude,
                      longitude, tlc, stanox, tiploc, station.
        min_km:       Minimum distance from closure centroid (default 10 km).
        max_km:       Maximum distance from closure centroid (default 25 km).

    Returns:
        DataFrame with one row per (closure, nearby station) pair.
    """
    station_lat = stations_df["latitude"].values
    station_lon = stations_df["longitude"].values

    tlc_to_stanox = stations_df.set_index("tlc")["stanox"].to_dict()
    tlc_to_tiploc = stations_df.set_index("tlc")["tiploc"].to_dict()

    rows = []
    for _, row in road_df.iterrows():
        lat, lon  = row["closure_lat"], row["closure_lon"]
        distances = haversine_vectorised(lat, lon, station_lat, station_lon)

        mask   = (distances >= min_km) & (distances <= max_km)
        nearby = stations_df[mask].copy()
        nearby["distance_km"] = distances[mask]

        for _, st in nearby.iterrows():
            tlc = st["tlc"]
            rows.append({
                "closure_id":               row.get("situation_id"),
                "closure_type":             row["closure_type"],
                "closure_start_time":       pd.to_datetime(row["start_time"]).tz_localize(None),
                "closure_end_time":         pd.to_datetime(row["end_time"]).tz_localize(None),
                "road_name":                row["road_name"],
                "road_class":               row["road_class"],
                "closure_severity":         row["closure_severity"],
                "start_hour":               row["start_hour"],
                "start_dow":                row["start_dow"],
                "start_date":               row["start_date"],
                "duration_hours":           row["duration_hours"],
                "effective_duration_hours": row["effective_duration_hours"],
                "effective_end_time":       pd.to_datetime(row["effective_end_time"]).tz_localize(None),
                "effective_start_time":     pd.to_datetime(row["effective_start_time"]).tz_localize(None),
                "validity_status":          row["validity_status"],
                "cause_type":               row["cause_type"],
                "station_name":             st["station_name"],
                "station_code":             tlc,
                "stanox":                   tlc_to_stanox.get(tlc),
                "tpl":                      tlc_to_tiploc.get(tlc),
                "distance_in_km":           st["distance_km"],
            })

    return pd.DataFrame(rows)