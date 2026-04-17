"""Centralised configuration — loads credentials from .env file."""

import os

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError(
        "Missing dependency: python-dotenv. Install it with `pip install -r requirements.txt`."
    ) from exc

load_dotenv()

# Azure Blob Storage
AZURE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
if AZURE_CONNECTION_STRING:
    AZURE_CONNECTION_STRING = AZURE_CONNECTION_STRING.strip().strip('"').strip("'")

if not AZURE_CONNECTION_STRING:
    raise RuntimeError(
        "AZURE_STORAGE_CONNECTION_STRING is not set. Copy `.env.example` to `.env` and add your Azure connection string, "
        "or set the environment variable before running the script."
    )

# Container names
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
