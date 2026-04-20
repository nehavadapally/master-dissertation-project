"""Geospatial utilities — haversine distance and closure-to-station matching."""

import numpy as np
import pandas as pd


def haversine_vectorised(lat1, lon1, lat2_array, lon2_array):
    """Vectorised haversine distance (km) between one point and arrays of points."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2_array), np.radians(lon2_array)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def find_nearby_stations(road_df, stations_df, stations_lookup_df,
                         min_km=10, max_km=25):
    """Expand road closures by matching each to stations within a distance band.

    Args:
        road_df: road closure DataFrame (needs closure_lat, closure_lon,
                 start_time, end_time, closure_type, validity_status, situation_id,
                 ingested_at columns).
        stations_df: station DataFrame (needs Station, TLC, Latitude, Longitude).
        stations_lookup_df: TIPLOC lookup with '3alpha', 'stanox', 'tiploc' columns.
        min_km: minimum distance from closure (default 10).
        max_km: maximum distance from closure (default 25).

    Returns:
        DataFrame with one row per (closure, nearby station) pair.
    """
    station_lat = stations_df["Latitude"].values
    station_lon = stations_df["Longitude"].values

    tlc_to_stanox = stations_lookup_df.set_index("3alpha")["stanox"].to_dict()
    tlc_to_tiploc = stations_lookup_df.set_index("3alpha")["tiploc"].to_dict()

    rows = []
    for _, row in road_df.iterrows():
        lat, lon = row["closure_lat"], row["closure_lon"]
        distances = haversine_vectorised(lat, lon, station_lat, station_lon)

        mask = (distances >= min_km) & (distances <= max_km)
        nearby = stations_df[mask].copy()
        nearby["distance_km"] = distances[mask]

        for _, st in nearby.iterrows():
            tlc = st["TLC"]
            rows.append({
                "closure_id": row.get("situation_id"),
                "closure_lat": lat,
                "closure_lon": lon,
                "closure_type": row["closure_type"],
                "closure_start_time": pd.to_datetime(row["start_time"]).tz_localize(None),
                "closure_end_time": pd.to_datetime(row["end_time"]).tz_localize(None),
                "validity_status": row["validity_status"],
                "station_name": st["Station"],
                "station_lat": st["Latitude"],
                "station_lon": st["Longitude"],
                "station_code": tlc,
                "stanox": tlc_to_stanox.get(tlc),
                "tpl": tlc_to_tiploc.get(tlc),
                "distance_in_km": st["distance_km"],
                "road_ingestion_time": pd.to_datetime(row["ingestion_time"]).tz_localize(None),
            })

    return pd.DataFrame(rows)
