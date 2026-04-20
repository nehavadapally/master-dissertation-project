# Rail Delay Prediction

Predicts rail service delays caused by nearby road closures using UK rail and road data from Azure Blob Storage.

## Project structure

```
rail-delay-prediction/
├── .env.example              # Credential template (copy to .env)
├── .gitignore
├── requirements.txt
├── README.md
├── test_connection.py        # Azure connection smoke test
├── src/
│   ├── __init__.py
│   ├── config.py             # Loads .env, defines container names & paths
│   ├── azure_client.py       # Azure Blob Storage ops + local file caching
│   ├── parsers.py            # CSV/JSON loader + Darwin timetable parser
│   ├── data_loader.py        # Downloads & loads all data sources
│   ├── geo.py                # Haversine distance & closure-to-station matching
│   └── features.py           # Timetable reshaping, merge/filter, delay calc
└── notebooks/
    └── rail_delay_prediction.ipynb   # EDA + modelling (the main analysis)
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env with your Azure Storage connection string

# 3. Verify connection
python test_connection.py

# 4. Run the analysis
cd notebooks
jupyter notebook rail_delay_prediction.ipynb
```

## Data sources

| Source | Azure container | Description |
|--------|----------------|-------------|
| Road closures | `road-closures` | Planned & unplanned road closure events |
| Train moments | `train-moments` | Real-time train movement records |
| Darwin timetable | `darwin-timetable-feeds` | Scheduled timetable (XML) |
| Station reference | `rail-road-data` | GB stations CSV + CORPUS TIPLOC lookup |

## Approach

1. **Geospatial matching** — each road closure is matched to rail stations within 10–25 km (haversine)
2. **Temporal filtering** — only train services within 60 minutes of a closure start are kept
3. **Feature engineering** — distance, time-since-closure, event type, interaction terms
4. **Modelling** — Linear Regression, Random Forest, Gradient Boosting, XGBoost
5. **Prediction** — best model predicts delays for future timetabled services near active closures
