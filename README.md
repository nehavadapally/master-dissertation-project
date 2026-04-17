## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Azure Storage connection string
```

## Project structure

```
src/
├── config.py          # Loads .env, defines container names and local paths
└── azure_client.py    # Azure Blob Storage client (download by time window or name)
```

## Usage

```python
from datetime import datetime, timezone
from src.azure_client import download_blobs_in_window, list_blobs
from src.config import CONTAINER_ROAD_CLOSURES, ROAD_DIR

# List blobs in a container
blobs = list_blobs(CONTAINER_ROAD_CLOSURES)

# Download blobs within a time window
start = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
end   = datetime(2026, 4, 12, 23, 59, 59, tzinfo=timezone.utc)

download_blobs_in_window(CONTAINER_ROAD_CLOSURES, ROAD_DIR, start, end)
```
