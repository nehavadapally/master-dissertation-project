"""Centralised configuration — loads credentials from .env file."""

import os
from dotenv import load_dotenv

load_dotenv()

AZURE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

# Azure container names
CONTAINER_ROAD_CLOSURES = "road-closures"
CONTAINER_TRAIN_MOMENTS = "train-moments"
CONTAINER_DARWIN_REALTIME = "darwin-realtime-feeds"
CONTAINER_DARWIN_TIMETABLE = "darwin-timetable-feeds"
CONTAINER_RAIL_ROAD_DATA = "rail-road-data"

# Local data directories
DATA_DIR = "data"
ROAD_DIR = os.path.join(DATA_DIR, "road")
RAIL_DIR = os.path.join(DATA_DIR, "rail")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
DARWIN_DIR = os.path.join(DATA_DIR, "darwin")
DARWIN_TIMETABLE_DIR = os.path.join(DATA_DIR, "darwin_timetable")
