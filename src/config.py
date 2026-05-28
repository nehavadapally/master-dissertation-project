"""Centralised configuration - loads credentials from .env file."""

import os
from dotenv import load_dotenv

# Load all environment variables from .env file into the application
load_dotenv()

# Road Closures API configuration
ROAD_CLOSURES_API_KEY = os.environ["ROAD_CLOSURES_API_KEY"]
ROAD_CLOSURES_URL = os.environ['ROAD_CLOSURES_URL']
ROAD_CLOSURE_HEADERS = {
    "Ocp-Apim-Subscription-Key": ROAD_CLOSURES_API_KEY,
    "X-Response-MediaType": "application/xml",
    "X-Data-Format": "DATEXII"
}

# Kafka configuration for real-time feeds
REAL_TIME_FEEDS_TOPIC = os.environ["REAL_TIME_FEEDS_TOPIC"]
REAL_TIME_FEEDS_CONFIG = {
    'bootstrap.servers': os.environ["KAFKA_BOOTSTRAP_SERVERS"],
    'security.protocol': os.environ["KAFKA_SECURITY_PROTOCOL"],
    'sasl.mechanism': os.environ["KAFKA_SASL_MECHANISM"],
    'sasl.username': os.environ["KAFKA_SASL_USERNAME"],
    'sasl.password': os.environ["KAFKA_SASL_PASSWORD"],
    'group.id': os.environ["REAL_TIME_FEEDS_GROUP_ID"],
    'auto.offset.reset': os.environ["KAFKA_AUTO_OFFSET_RESET"]
}

# Train Moments Kafka configuration
TRAIN_MOMENTS_TOPIC = os.environ["TRAIN_MOMENTS_TOPIC"]
TRAIN_MOMENTS_CONFIG = {
    'bootstrap.servers': os.environ["KAFKA_BOOTSTRAP_SERVERS"],
    'security.protocol': os.environ["KAFKA_SECURITY_PROTOCOL"],
    'sasl.mechanism': os.environ["KAFKA_SASL_MECHANISM"],
    'sasl.username': os.environ["KAFKA_SASL_USERNAME"],
    'sasl.password': os.environ["KAFKA_SASL_PASSWORD"],
    'group.id': os.environ["TRAIN_MOMENTS_GROUP_ID"],
    'auto.offset.reset': os.environ["KAFKA_AUTO_OFFSET_RESET"]
}

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