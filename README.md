# Rail Delay Prediction
### Road–Rail Resilience: Early Warning Model for Multi-Modal Disruption
**Kainos MSc Dissertation 2026**

Predicts the probability of rail service delays caused by nearby road closures on the Strategic Road Network (SRN), using real-time and scheduled UK road and rail data ingested from Azure Blob Storage.

---

## Project Structure

```
rail-delay-prediction/
├── .env.example                        # Credential template (copy to .env)
├── .gitignore
├── venv
├── requirements.txt
├── README.md
├── test_connection.py                  # Azure connection smoke test
├── src/
│   ├── __init__.py
│   ├── config.py                       # Loads .env, defines container names & local paths
│   ├── azure_client.py                 # Azure Blob Storage ops + local file caching
│   ├── parsers.py                      # CSV/JSON loader + Darwin timetable parser
│   ├── data_loader.py                  # Downloads & loads all data sources
│   ├── geo.py                          # Haversine distance & closure-to-station matching
│   └── features.py                     # Timetable reshaping, merge/filter, delay calc
└── notebooks/
    ├── data_ingestion/
    │   ├── darwin_realtime_data.ipynb  # Darwin real-time feed ingestion to Azure
    │   ├── road_closures_data.ipynb    # DATEX II road closure ingestion to Azure
    │   └── train_moments_data.ipynb    # Network Rail TRUST Kafka stream to Azure
    ├── eda_01_stations_reference.ipynb # GB stations + CORPUS crosswalk EDA
    ├── eda_02_road_closures.ipynb      # Road closure EDA (296 records, 72h window)
    ├── eda_03_train_moments.ipynb      # Train moments EDA (41,026 raw records)
    ├── eda_04_darwin_timetable.ipynb   # Darwin timetable EDA (1.9M stop rows)
    ├── eda_05_road_train_moments_dataset.ipynb  # Merged retrospective dataset (5,446 rows)
    ├── eda_06_road_timetable_dataset.ipynb      # Merged prediction dataset (100,069 rows)
    └── modelling.ipynb     # Modelling pipeline (main analysis)
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env with your Azure Storage connection string and Kafka credentials

# 3. Verify connection
python test_connection.py

# 4. Run EDA notebooks in order (eda_01 → eda_06), then the modelling notebook
cd notebooks
jupyter notebook modelling.ipynb
```

---

## Data Sources

| Source | Azure Container | Format | Records (72h window) |
|--------|----------------|--------|----------------------|
| Road closures (DATEX II) | `road-closures` | XML → CSV | 296 closures |
| Train moments (TRUST) | `train-moments` | Kafka → CSV | 41,026 raw / 39,091 clean |
| Darwin timetable | `darwin-timetable-feeds` | XML → JSON | ~1.9M stop rows |
| GB stations reference | `rail-road-data` | CSV | 2,595 stations |
| CORPUS TIPLOC lookup | `rail-road-data` | JSON | 55,920 records |

All sources are open or licensed under Rail Data Marketplace terms. No personally identifiable information is processed at any stage.

---

## Approach

### 1. Data Ingestion
Three Kafka/REST ingestion notebooks stream road closure, Darwin real-time, and train moment data into Azure Blob Storage as timestamped CSV files. A local caching layer in `azure_client.py` avoids re-downloading blobs already present on disk, with time-window filtering via filename-encoded timestamps.

### 2. Identifier Crosswalk
Train moments carry STANOX codes; the Darwin timetable uses TIPLOC codes; the station reference is indexed by TLC (3-letter code). The CORPUS extract (`CORPUSExtract.json`, 55,920 records) provides the STANOX → TIPLOC → 3ALPHA mapping that links all three sources. Match rate: 99.96% of stations (2,594 / 2,595).

### 3. Geospatial Matching
Each road closure centroid is matched to rail stations within a **10–25 km haversine band** using `geo.find_nearby_stations()`. The lower bound excludes stations with direct physical proximity to the closure; the upper bound reflects the hypothesised attenuation of cross-modal modal shift effects. Applied to 296 closures, this produces 16,361 closure–station pairs across the observation window.

### 4. Temporal Filtering
Only rail service events whose planned timestamp falls **within 60 minutes of a closure start** are retained, implemented in `features.filter_within_time_window()`. Records are uniformly distributed across 10-minute sub-buckets within the window, confirming no systematic temporal bias.

### 5. Analytical Datasets

Two datasets are produced after spatial join and temporal filtering:

| Dataset | Source | Rows | Closures | Stations | Purpose |
|---------|--------|------|----------|----------|---------|
| Retrospective | Train moments | 5,446 | 126 | 854 | Model training (delay label known) |
| Forward-looking | Darwin timetable | 100,069 | 190 | 1,467 | Prediction (planned timestamps only) |

The timetable dataset is ~18.4× larger than the retrospective dataset. 621 stations and 64 closures appear only in the prediction set, representing a deliberate generalisation challenge for the model.

### 6. Feature Engineering

Features used in the model, implemented in `features.py`:

| Feature | Type | Description |
|---------|------|-------------|
| `distance_in_km` | Continuous | Haversine distance from closure centroid to station (10–25 km) |
| `planned_time_diff` | Continuous | Minutes from closure start to planned service (0–60 min) |
| `distance_time_interaction` | Continuous | distance × planned_time_diff |
| `start_hour` | Categorical | Hour of day of closure start |
| `start_dow` | Categorical | Day of week of closure start |
| `closure_type` | Categorical | planned / unplanned |
| `cause_type` | Categorical | roadMaintenance, accident, etc. |
| `validity_status` | Categorical | active / suspended / planned |
| `lanes_closed` | Numeric | Proxy for disruption severity (0–4) |
| `event_type` | Categorical | ARRIVAL / DEPARTURE / PASS |



### 7. Modelling

Four candidate models evaluated on the retrospective dataset with a temporal train/test split:

- Linear Regression (interpretable baseline)
- Random Forest
- Gradient Boosting
- XGBoost

Primary evaluation metrics: ROC-AUC and F1 score (appropriate given class imbalance). The best-performing model is applied to the 100,069-row forward-looking timetable dataset to produce per-service delay probability scores, ranked by predicted risk.

---

## Key EDA Findings

| Finding | Value |
|---------|-------|
| Observation window | 10–13 April 2026 (72 hours) |
| Road closures | 352 total — 214 planned, 138 unplanned |
| Train moments (clean) | 39,091 rows — 38.1% delayed (variation_status) |
| Raw delay distribution | Median 1 min, mean 2.53 min, max 293 min (skewness 12.31) |
| Timetable stop rows | ~1.9M across ~121K journeys |
| Retrospective dataset | 93,749 rows, 184 closures, 1373 stations |
| Prediction dataset | 269,497 rows, 243 closures, 1,566 stations |
| Pearson r (distance vs delay) | –0.010 |
| Pearson r (time diff vs delay) | -0.002 |

The weak linear correlation between spatial/temporal proximity and delay motivates a non-linear classification approach rather than regression.

---

## Requirements

```
azure-storage-blob>=12.0
confluent-kafka
pandas>=2.0
numpy>=1.24
geopandas
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
python-dotenv>=1.0
pyarrow>=15.0.0
lxml
```

---

## Notes

- The `data/` directory is excluded from version control (`.gitignore`). All intermediate parquet files are regenerated by running EDA notebooks in order.
- Credentials are never stored in source copy `.env.example` to `.env` and populate before running.
- The pipeline is designed for extensibility: Street Manager emergency works and BODS bus feeds are the next planned data source integrations.