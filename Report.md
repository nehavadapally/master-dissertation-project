# Road-Rail Resilience: Technical Report

## 1. Introduction

This project examines whether unplanned road closures on the Strategic Road Network and Major Road Network can help predict rail performance degradation at nearby stations. The starting point is a practical operational problem. Road and rail systems are experienced by passengers as connected networks, but the data used to manage them is usually held in separate systems. National Highways records road closures through DATEX II feeds, while Network Rail and the Rail Delivery Group publish rail movement and timetable data through TRUST and Darwin. These sources are valuable individually, but they do not naturally support cross-modal analysis.

The aim of this project is to build a proof-of-concept machine learning model that predicts whether a rail station will experience degraded performance on a given day when road closures occur within a 10 to 25 kilometre radius. A station-day is defined as disrupted when mean arrival delay is greater than five minutes. The work is not designed to prove causality. The available data cannot show that road closures directly cause rail delays. Instead, the project tests whether road closure features contain enough predictive signal to support an early warning tool.

## 2. Data and Infrastructure

The project integrates five datasets. DATEX II road closure data provides the external disruption signal. TRUST train movement data provides the rail performance outcome. Darwin timetable data supports forward prediction because it contains planned services that have not yet operated. GB Stations provides station coordinates. CORPUS links identifiers across rail datasets by mapping STANOX, TIPLOC and CRS codes.

The ingestion and modelling layers are separated. Google Colaboratory was used for ingestion because TRUST and DATEX II required ongoing data collection. Azure Blob Storage was used as the shared data layer. The modelling and dashboard work were completed in Python using Pandas, NumPy, scikit-learn, XGBoost, LightGBM, Folium, Matplotlib and Gradio.

```mermaid
erDiagram
    STATIONS_REFERENCE {
        string station PK
        string tlc
        string stanox
        string tiploc
        float latitude
        float longitude
    }

    ROAD_CLOSURES_CLEAN {
        string situation_id PK
        string closure_type
        string road_name
        int road_class
        float closure_severity
        float closure_lat
        float closure_lon
        float effective_duration_hours
        string cause_type
    }

    TRAIN_MOVEMENTS_CLEAN {
        string train_id PK
        string station_name FK
        date planned_date
        float delay_minutes
        int is_late
        int is_delayed_5min
    }

    DARWIN_TIMETABLE {
        string rid PK
        string tpl FK
        date ssd
        string wta
        string wtd
    }

    ROAD_STATION_DAY_DATASET {
        string station_name PK
        date planned_date PK
        int train_movements
        float mean_delay_minutes
        int has_road_closure
        int road_closure_count
        float min_distance_km
        float inv_distance_sum
        float total_closure_severity
        int closures_lag7d
        int is_friday
        int disrupted
    }

    ROAD_TIMETABLE_STATION_DAY {
        string station_name PK
        date planned_date PK
        int train_movements
        float disruption_probability
        string risk_band
        float predicted_delay_minutes
        int has_road_closure
        float total_closure_severity
        int closures_lag7d
    }

    STATIONS_REFERENCE ||--o{ TRAIN_MOVEMENTS_CLEAN : "stanox resolves station"
    STATIONS_REFERENCE ||--o{ ROAD_STATION_DAY_DATASET : "haversine spatial join"
    ROAD_CLOSURES_CLEAN ||--o{ ROAD_STATION_DAY_DATASET : "closure features aggregated"
    TRAIN_MOVEMENTS_CLEAN ||--|| ROAD_STATION_DAY_DATASET : "station-day aggregation"
    DARWIN_TIMETABLE ||--o{ ROAD_TIMETABLE_STATION_DAY : "TIPLOC resolved, joined"
    ROAD_CLOSURES_CLEAN ||--o{ ROAD_TIMETABLE_STATION_DAY : "forward closure features"
    STATIONS_REFERENCE ||--o{ DARWIN_TIMETABLE : "tpl = tiploc"
```

The repository is organised around three main areas:

| Area | Purpose |
|---|---|
| `notebooks/data_ingestion/` | Collects road, rail and timetable feeds |
| `notebooks/eda_*.ipynb` | Cleans, joins and analyses the datasets |
| `src/` | Supports the Gradio application, maps, charts and model outputs |

The dashboard is launched through `app.py`. It loads the interface layout from `src/layout.py`, data and model artefacts from `src/data.py`, maps from `src/maps.py` and summary charts from `src/dashboard.py`.

## 3. Pipeline Design

The final pipeline works at station-day level. Earlier work considered train-level prediction, but this created too much sparsity and noise. A single road closure can affect many services differently and TRUST does not contain passenger loading or crowding information. Station-day aggregation gives a more stable representation of local rail performance.

```mermaid
flowchart LR
    A[DATEX II closures] --> B[Clean road events]
    C[TRUST movements] --> D[Aggregate station-day delay]
    E[Darwin timetable] --> F[Forward station-day schedule]
    G[GB Stations and CORPUS] --> H[Station reference]
    B --> I[10 to 25 km spatial join]
    H --> I
    D --> J[Model training dataset]
    I --> J
    J --> K[XGBoost classifier]
    J --> L[XGBoost regressor]
    F --> M[Forward predictions]
    K --> M
    L --> M
    M --> N[Gradio dashboard]
```

The spatial join uses haversine distance between each road closure centroid and each station coordinate. Only stations between 10 and 25 kilometres from a closure are retained. The lower bound reduces the risk of capturing direct infrastructure proximity effects. The upper bound reflects the assumption that passenger diversion becomes less plausible as distance increases.

Road closures are expanded across the dates on which they are active. Where more than one closure affects the same station-day, cumulative features are calculated. This allows the model to capture the combined effect of multiple nearby road events.

## 4. Feature Engineering

Features are organised into five groups.

| Group | Example Features |
|---|---|
| Closure attributes | Closure count, unplanned closure count, minimum distance and duration |
| Spatial severity | Road class severity and inverse-distance weighted closure score |
| Temporal memory | One-day, three-day and seven-day lag closure counts |
| Calendar effects | Day of week, Monday flag, Friday flag and weekend flag |
| Rail volume | Train movement count at station-day level |

The main target is binary. A station-day is labelled disrupted when mean arrival delay is greater than five minutes. This threshold gives the prediction a practical interpretation. It identifies station-days where performance degradation is large enough to matter operationally.

A complementary regression target estimates mean delay in minutes. This supports the dashboard by showing both disruption probability and expected delay magnitude.

## 5. Dataset Overview

The final integrated station-day dataset contains 33,941 rows from 3 April to 28 April 2026. It includes 2,506 stations, 1,368 cleaned road closures and 212,884 cleaned TRUST movement records. Of the station-days, 20,052 had at least one road closure active within the 10 to 25 kilometre radius. A smaller subset of 15,040 had at least one unplanned closure nearby.

The five-minute threshold produced 1,916 disrupted station-days and 32,025 non-disrupted station-days. This gives a disruption rate of 5.65% and a class imbalance ratio of 16.7:1. The imbalance shaped the evaluation approach. Accuracy was not used as the main metric because a model could achieve high accuracy by predicting no disruption for almost every row.


## 6. Exploratory Findings

The correlation analysis showed that the cross-modal signal is weak at station-day level. No feature had an absolute Pearson correlation above 0.09 with mean arrival delay. Road closure count, unplanned closure count, inverse-distance score and total closure severity all had correlations below 0.02.

Day-of-week effects were stronger than most road features. Friday had the highest mean delay and highest disruption rate. This suggests that background rail operating patterns explain more variation than road closure features within the current dataset.

<p align="center">
  <img src="notebooks/figures/eda_05/correlation_matrix_full.png" width="600">
</p>

Closure-hour analysis provided a directional but not conclusive result. Station-days affected by unplanned closures showed slightly higher delays than station-days affected only by planned closures. This is operationally plausible because unplanned events give operators less time to respond. However, the subset is too small to support a strong claim.

## 7. Modelling Results

Six classification models were evaluated using a chronological train-test split. The main model was XGBoost, with Logistic Regression, Random Forest, Gradient Boosting, LightGBM and a dummy baseline used for comparison. Precision-Recall AUC was selected as the primary metric because disrupted station-days are rare.

<p align="center">
  <img src="notebooks/figures/classification/model_comparison.png" width="600">
</p>

XGBoost achieved the highest PR-AUC at 0.120 against a random baseline of 0.065. This is an improvement over chance, but it remains weak for operational use. At the selected threshold of 0.53, the model recovered 49% of disrupted station-days but produced low precision. In practice, this means many false alerts would be generated.

<p align="center">
  <img src="notebooks/figures/classification/precision_recall_curves.png" width="600">
</p>

The most important classification features were maximum effective closure duration, train movement count, Friday indicator, total closure severity and seven-day closure lag. The binary feature showing whether any road closure existed had zero importance once severity and duration features were included. This is important because it shows that the model responds to closure magnitude rather than simple closure presence.

<p align="center">
  <img src="notebooks/figures/classification/feature_importance.png" width="600">
</p>

The regression model gave weaker results. Tuned XGBoost achieved an MAE of 1.552 minutes, only slightly better than the dummy mean baseline at 1.576 minutes. The R2 score was 0.010, meaning the model explained approximately 1% of delay variation. This confirms that delay magnitude remains difficult to predict with the available open data features.

<p align="center">
  <img src="notebooks/figures/regression/model_comparison.png" width="600">
</p>

## 8. Dashboard Output

The Gradio dashboard provides an operational view of the pipeline. Users can filter closures by date, time, closure type, distance and duration. Selecting a closure shows nearby stations. Selecting a station shows predicted disruption probability, predicted delay, historical performance and timetable prediction charts.

<p align="center">
  <img src="notebooks/figures/gradio_01.png" width="600">
</p>
<p align="center">
  <img src="notebooks/figures/gradio_llm_01.png" width="600">
</p>


The dashboard also includes a data overview page with dataset metrics, model performance and feature importance. This is useful for explaining the model limitations to non-technical users.


<p align="center">
  <img src="notebooks/figures/gradio_overview_01.png" width="600">
</p>

<p align="center">
  <img src="notebooks/figures/gradio_overview_02.png" width="600">
</p>

## 9. Discussion

The project successfully built a reusable open-data pipeline for road-rail integration. This is the strongest contribution. The model results are weaker, but still useful. They show that the signal exists above chance but is not strong enough for deployment.

The main limitation is not the model choice. It is the data. Eighteen days of spring data is too short to capture seasonal disruption patterns. Station-day aggregation may hide peak-hour effects. CORPUS does not resolve every rail location. Road closure centroids simplify long closures and can distort distance features. Most importantly, there is no open station-level passenger volume dataset. Without passenger demand, the model cannot distinguish a road-induced demand surge from normal variation, events, holidays or within-rail causes.

## 10. Conclusion

This project demonstrates that UK open road and rail data can be integrated into a coherent modelling pipeline. It also shows that the current station-day feature set provides only a weak predictive signal. The XGBoost classifier performs above chance, but its precision is too low for operational deployment. The regression model explains very little variation in delay magnitude.

The main conclusion is therefore balanced. The pipeline is technically successful and academically useful. The current predictive performance is limited. Future work should extend the data collection window, move to station-hour modelling and add passenger volume or observed traffic flow features. These additions would give the cross-modal hypothesis a stronger and fairer test.
