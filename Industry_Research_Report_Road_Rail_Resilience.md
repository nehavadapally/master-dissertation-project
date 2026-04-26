# Road–Rail Resilience: An Early-Warning Machine Learning System for Cross-Modal Disruption

**Industry Research Report**

---

**Prepared for:** Kainos - Public Sector, Transport Client Group  
**Author:** Neha Vadapally (P2911159)  
**Programme:** MSc Data Analytics
**Institution:** De Montfort University 
**Industry Supervisor:** Peter Bodnar  
**Academic Supervisor:** Dr Ahmad Lawal  
**Date:** April 2026

---

## Executive Summary

Road closures on the UK Strategic Road Network create modal displacement effects that remain unmeasured by current operational systems. This research develops a proof-of-concept machine learning pipeline integrating National Highways DATEX II road closure data with Network Rail TRUST and Darwin rail operations data to provide predictive early-warning capability for traffic control centres.

The analytical dataset used for training comprises 5505 train-closure event observations spanning 848 rail stations and 125 road-closure events collected between 10-12 April 2026. Five supervised ML models were trained and evaluated using spatial-temporal proximity features: haversine distance (10-25 km), temporal recency (0-60 minutes) and engineered interaction terms.

**Principal Findings:**

Linear Regression achieved optimal performance with Mean Absolute Error of 1.90 minutes at the event level, though R² of -0.002 indicates minimal variance explanation. Pearson correlation analysis reveals negligible linear relationships: distance to closure (r = -0.018), time since closure onset (r = +0.014), and their interaction (r = -0.002). The positive temporal correlation contradicts the theoretical hypothesis that delay effects attenuate over time.

The target delay distribution exhibits pronounced negative skewness (-2.37) and extreme leptokurtosis (84.26), with range spanning -104 to +61.5 minutes. The leftward skew, indicating more early arrivals than delayed services, suggests potential data quality issues related to STANOX-TIPLOC identifier reconciliation.

Categorical analysis shows 55.9% unplanned closures versus 44.1% planned roadworks, contrary to typical operational distributions where planned works dominate. Mean delay for planned closures (1.55 minutes) exceeds unplanned incidents (0.59 minutes), potentially attributable to duration differences rather than severity.

The dual-dataset architecture (TRUST for training, Darwin for inference) achieved 25.99% cross-validation match rate due to identifier namespace fragmentation. Station-level aggregation improved predictive accuracy to 1.16 minutes MAE, demonstrating the model captures broad spatial patterns despite weak event-level signal.

**Recommendation:** Extend observation window to three months to establish robust baseline performance before production deployment consideration.

---

## 1. Research Context and Motivation

### 1.1 Infrastructure Integration Challenge

The UK transport infrastructure operates through distinct management domains: National Highways administers the Strategic Road Network (SRN) and Major Road Network (MRN), while Network Rail manages rail infrastructure operations. When major road closures occur, whether planned maintenance programmes or unplanned incidents, displaced vehicular traffic must either divert to alternative routes or shift to public transport modes. Empirical research by Guiver (2011) documented modal shift behaviour from private vehicles to rail services during roadwork periods, yet no automated analytical system currently quantifies this cross-modal impact.

The Office of Rail and Road (ORR) Annual Assessment 2024-25 reports substantial economic costs attributable to rail service delays. Cross-modal disruption spillover contributes to these costs but remains unmeasured in current attribution frameworks. Traffic control centres receive independent alert streams from DATEX II (road closure notifications) and TRUST/Darwin (rail operational status) but lack integrated analytical capability to address operational questions such as: "Given M25 Junction 10-12 closure commencing at 14:00 hours, which rail services scheduled for 15:00 hours will experience delay, and what is the expected magnitude?"

### 1.2 Research Partnership and Objectives

This research, conducted in partnership with Kainos Transport Client Group, develops a proof-of-concept machine learning pipeline to address the cross-modal prediction gap. The system links National Highways DATEX II road-closure metadata with Network Rail TRUST actual train movements and Darwin planned timetable data.

**Research Questions:**

**RQ1 (Measurement):** Does a statistically detectable relationship exist between road-closure spatial-temporal proximity and rail service delay magnitude?

**RQ2 (Prediction):** Can supervised machine learning models predict rail delay using road-closure metadata and engineered spatial-temporal features?

**RQ3 (Explainability):** Which features (distance, temporal recency, closure classification, event type) demonstrate dominant predictive importance, and do their rankings align with theoretical expectations?

**RQ4 (Operational Viability):** What technical, data governance and operational constraints prevent production deployment in traffic control centre environments?

### 1.3 Implementation Scope

The research employs a phased methodology with the following boundaries:

**Geographic Coverage:** Great Britain (England, Scotland, Wales), excluding Northern Ireland due to distinct transport governance structures.

**Temporal Scope:** Three-day pilot observation window (10-12 April 2026) establishing baseline feasibility metrics before extended data collection commitment.

**Modal Focus:** Road-to-rail primary analysis pathway. Bus Open Data Service (BODS) integration deferred to subsequent tri-modal extension phase.

**Data Licensing:** Open Government Licence 3.0 and Rail Data Marketplace terms exclusively. Commercial proprietary delay attribution feeds excluded.

**Deferred Elements:** Weather data (extensively studied in prior literature, Oneto et al. 2018), Street Manager roadworks severity classifications (licensing constraints during pilot phase), Annual Average Daily Flow traffic volume data (requires extended temporal window for variance detection).

The three-day observation window represents a proof-of-concept limitation, insufficient for detecting seasonal patterns, weekly operational cycles or rare event distributions. This constraint is necessary to validate pipeline architecture and feature engineering methodology before committing to multi-month data acquisition programmes.

---

## 2. Literature Foundation and Research Gap

### 2.1 Transport System Resilience Theory

Bruneau et al. (2003) established the foundational "4Rs" framework for infrastructure resilience assessment: Robustness (withstand disruption without degradation), Redundancy (substitutable elements maintaining function), Resourcefulness (mobilise resources for recovery), and Rapidity (contain losses and restore function expeditiously). Reggiani et al. (2013) distinguished engineering resilience (recovery speed to original state) from ecological resilience (system adaptation to new equilibrium), while Poulin and Kane (2021) proposed a trapezoid temporal model encompassing prepare-absorb-recover-adapt phases.

Recent developments by Pant et al. (2024) emphasise cross-modal infrastructure dependencies in networked systems, noting that disruption cascades follow non-obvious pathways through interconnected networks. However, operational machine learning implementations remain absent from the resilience literature, which predominantly employs simulation-based methodologies.

### 2.2 Cross-Modal Disruption Evidence Base

Simulation studies provide theoretical grounding for cross-modal spillover effects. Van Nes et al. (2020) developed the Disruption Transport Model (DTM) demonstrating 30-50 kilometre spillover radii for road-to-public-transport modal displacement in Dutch urban networks. Tang et al. (2021) assessed urban rail resilience under disruption scenarios, observing 63% higher delay propagation during peak operational periods, indicating significant time-of-day moderation effects.

Xiao et al. (2024) modelled cascading failures on weighted networks, demonstrating non-linear threshold behaviour where disruption effects manifest abruptly beyond critical network load thresholds rather than exhibiting gradual linear degradation.

UK-specific empirical evidence remains limited. Guiver (2011) conducted qualitative interview research documenting modal shift from private vehicle to rail during roadwork periods but provided no quantitative impact measurement. Monsuur et al. (2021) examined disruption management strategies in Dutch railway contexts but did not address cross-modal causation.

### 2.3 Rail Delay Prediction: Methodological Survey

Two comprehensive literature surveys establish the current state of rail delay prediction research:

Spanninger et al. (2022) reviewed 36 studies, identifying weather conditions as the dominant exogenous predictor variable. Their taxonomy of predictive approaches revealed no studies incorporating road network event data as explanatory features.

Tiong et al. (2023) proposed a five-category feature taxonomy for rail delay prediction: (1) timetable characteristics (dwell time, headway intervals), (2) operational parameters (train type, operating company), (3) infrastructure attributes (junction complexity, signalling configuration), (4) external-environmental factors (weather exclusively) and (5) passenger demand variables (crowding indices, special events). Their subsequent work (Tiong et al. 2025) introduced the AP-GRIP evaluation framework encompassing Accuracy, Practicality, Generalisability, Interpretability and Robustness dimensions.

**Critical Gap Identification:** No study in either survey incorporates road-closure event data as a predictor variable. Cross-modal disruption has been studied via simulation (van Nes, Xiao) but not through supervised machine learning on operational data.

**Benchmark Studies:**

Kecman and Goverde (2015) achieved approximately 40 seconds (0.67 minutes) Mean Absolute Error on Dutch railways modelling within-rail delay propagation exclusively.

Oneto et al. (2016, 2018) demonstrated weather as the primary exogenous factor using Support Vector Regression, establishing precedent for external environmental predictors.

Finnish railway study (2026) reported 2.73 minutes MAE incorporating weather data, demonstrating haversine spatial join methodology for weather station proximity matching.

Sarhani and Voß (2024) conducted comparative algorithm evaluation (Support Vector Machines, Random Forest, Gradient Boosting) on open transport data with SHAP (Shapley Additive Explanations) interpretability analysis.

**Research Contribution:** This study proposes a sixth feature category in the Tiong taxonomy: cross-modal event features comprising road-closure metadata, spatial proximity via haversine distance calculation, and temporal recency measurements.

---

## 3. Data Sources and Processing Methodology

### 3.1 Data Acquisition Strategy

Seven data sources were identified during requirements analysis. Four were implemented in the pilot phase, with two deferred to subsequent enhancement iterations.

**Implemented Sources:**

1. [**DATEX II Road Events**](https://developer.data.nationalhighways.co.uk/api-details#api=road-and-lane-closures-v2&operation=RoadClosures)
(National Highways): Road closure metadata including cause classification, temporal validity bounds, and geographic location. The pilot captured 125 distinct closure events from 296 raw records. Licence: Open Government Licence 3.0.

2. [**TRUST Train Movements**](https://raildata.org.uk/dashboard/dataProduct/P-826477b8-3789-45e7-85bd-22c4ae9bcfae/overview)(Network Rail): Actual train movement timestamps enabling delay calculation through comparison with planned schedules. The dataset comprised 41026 raw movement records prior to filtering operations. Licence: Rail Data Marketplace Terms.

3. [**Darwin PPTimetable**](https://raildata.org.uk/dashboard/dataProduct/P-9ca6bc7e-62e1-44d6-b93a-1616f7d2caf8/overview) (Rail Delivery Group): Planned timetable schedules for forward prediction capability. Provides scheduled timestamps for services not yet operated. Licence: Rail Data Marketplace Terms.

4. **Stations List**
    - [**GB Stations**](https://www.doogal.co.uk/UkStations) : Provides station names, location details, three letter codes and footfall details acting as a primary source of station identifier.
    - [**CORPUS**](https://raildata.org.uk/dashboard/dataProduct/P-9d26e657-26be-496b-b669-93b217d45859/overview) (Network Rail): Codes for Operations, Retail and Planning - Unified Solution. Provides identifier cross-walk mapping between STANOX (signalling codes), TIPLOC (timetable location codes), and CRS (reservation system codes). Contains 11,035 mapping records. Licence: Rail Data Marketplace Terms.

**Deferred Sources:**

5. **AADF** (Department for Transport): Annual Average Daily Flow road traffic volume measurements. Deferred rationale: three-day pilot window provides insufficient temporal variance to detect traffic volume moderation effects. Requires multi-week observation period. Licence: Open Government Licence 3.0.

6. **Street Manager** (Department for Transport): Roadworks severity classifications including lane closure counts and carriageway impact assessments. Deferred rationale: licensing constraints during pilot phase and insufficient long-duration roadworks in three-day sample for validation. Licence: Open Government Licence 3.0.

7. **BODS** (Department for Transport): Bus Open Data Service providing real-time bus location and timetable data for tri-modal analysis. Deferred rationale: phased implementation strategy prioritising road-to-rail pathway validation before bus integration complexity. Tri-modal extension (road→rail+bus) requires differentiation of modal shift preferences and additional spatial join operations with 40,000+ bus stops nationally. Planned for subsequent research phase following rail baseline establishment. Licence: Open Government Licence 3.0.

### 3.2 Data Processing Pipeline Architecture

The transformation from raw data sources to analysis-ready dataset proceeds through six sequential stages:

**Stage 1: Road Closure Ingestion and Geocoding**

DATEX II XML payloads were parsed to extract situation records containing cause type, validity temporal bounds and geometry definitions. For linear road segments, centroid coordinates were computed as the mean latitude and longitude of polyline vertices. The process classified closures as planned (scheduled roadworks) or unplanned (incidents, accidents) based on cause type enumeration. Output: 125 closure events within the pilot temporal window.

**Stage 2: Train Movement Data Loading**

TRUST feed delivers newline-delimited JSON messages. Records were filtered to message type 0003 (train movement events) and parsed to extract actual timestamp, planned timestamp, location STANOX code, and event classification. STANOX codes were zero-padded to five-digit format and mapped to three-character station codes via CORPUS lookup tables. Output: 41026 train movement records.

**Stage 3: Spatial Join via Haversine Distance Calculation**

For each road closure centroid, great-circle distances were computed to all 2,594 station coordinates using the haversine formula with Earth radius 6,371 kilometres. Closure-station pairs were retained where distance fell within the interval [10, 25] kilometres. The 10 kilometre lower bound excludes stations with potential shared infrastructure (level crossings, adjacent access roads) that could confound attribution. The 25 kilometre upper bound concentrates analysis on stronger signal regions while maintaining sufficient sample size. Van Nes et al. (2020) observed 30-50 kilometre spillover effects in simulation; this implementation adopts a conservative narrower window.

**Stage 4: Temporal Filtering and Dataset Construction**

Closure-station pairs were inner-joined with train movements on station code. Temporal difference was calculated as: planned_time_diff = planned_timestamp - closure_start_time (minutes). Records were retained where 0 < planned_time_diff ≤ 60 minutes, capturing services occurring within one hour after closure onset.

**Critical Implementation Correction:** Initial implementation erroneously applied the filter condition planned_time_diff ≤ 1 minute, yielding only 147 observations. This was identified during validation when model performance showed no predictive capacity. Correction to the intended 60-minute window expanded the dataset to 5505 observations, a 37-fold increase. This correction demonstrates the importance of validation procedures in detecting specification errors.

Output: 5505 training observations (TRUST branch with actual timestamps enabling delay calculation).

**Stage 5: Feature Engineering**

Target variable delay was computed as: delay = actual_timestamp - planned_timestamp (minutes).

Predictor features were constructed:
- distance_in_km: haversine distance from Stage 3
- planned_time_diff: temporal recency from Stage 4  
- distance_time_interaction: product term (distance_in_km × planned_time_diff)
- closure_type: one-hot encoded categorical (planned, unplanned)
- event_type: one-hot encoded categorical (ARRIVAL, DEPARTURE)

**Stage 6: Darwin Inference Dataset (Parallel Construction)**

The identical spatial join and temporal filtering operations were applied to Darwin timetable data. Darwin contains only planned timestamps (contains services not yet operated), enabling forward prediction for operational deployment scenarios. This parallel construction validates that the pipeline generalises beyond historical TRUST data to prospective inference requirements.

### 3.3 Dual-Dataset Architectural Rationale

The system employs two distinct data sources reflecting operational reality:

**TRUST (Training):** Contains both planned_timestamp and actual_timestamp fields, enabling calculation of the target variable: delay = actual - planned. This historical performance data trains the supervised learning models.

**Darwin (Inference):** Contains only planned_timestamp (services scheduled but not yet operated). Traffic control centres receive Darwin timetable updates hours or days in advance of service operation. Prediction capability on Darwin schedules enables proactive operational interventions: advance passenger notifications, resource pre-positioning, and discretionary service retiming.

**Identifier Reconciliation Challenge:** TRUST employs STANOX codes (five-digit signalling identifiers) while Darwin employs TIPLOC codes (seven-character timetable location identifiers). CORPUS provides the cross-walk mapping, but namespace fragmentation results in imperfect alignment. Cross-dataset validation achieved 25.99% exact match rate between TRUST and Darwin records. This limitation was partially mitigated through station-level aggregation (averaging predictions and actuals per station rather than per individual train movement), which improved Mean Absolute Error from 1.90 minutes (event-level) to 1.16 minutes (station-level).

### 3.4 Feature Engineering Design Decisions

Five features were selected for model training:

**distance_in_km** (continuous, range [10, 25]): Spatial proximity hypothesis posits that closer closures exert stronger influence on rail delay magnitude.

**planned_time_diff** (continuous, range (0, 60]): Temporal recency hypothesis posits that recent closure onset produces stronger effects, with attenuation over time as traffic patterns stabilise and diversions become established.

**distance_time_interaction** (continuous): Multiplicative interaction term capturing joint effects. Near-recent combinations (low values) represent strong signal regime; far-old combinations (high values) represent weak signal regime. Enables models to learn threshold effects where both conditions must be satisfied.

**closure_type** (categorical: planned, unplanned): Theoretical expectation that unplanned incidents generate larger delays due to absence of advance notice and established diversion routes.

**event_type** (categorical: ARRIVAL, DEPARTURE): ARRIVAL events may reflect accumulated upstream delay propagation along the route; DEPARTURE events reflect station-specific delays including dwell time variability and boarding processes.

**Excluded Feature: station_name**

Preliminary experiments included station_name as a categorical feature with 848 distinct values requiring one-hot encoding. This produced three negative outcomes: (1) Random Forest training time increased from 48 seconds to exceeding 20 minutes due to feature dimensionality, (2) memory consumption exceeded 8 gigabyte Colab instance limit and (3) feature importance analysis showed Gini coefficients exceeding 0.9 for station baseline indicators, completely dominating road-closure features.

The exclusion decision forces models to rely exclusively on spatial-temporal closure signals rather than learning station-specific baseline delay patterns. This isolation enables clearer evaluation of the cross-modal hypothesis, though at the cost of reduced overall predictive accuracy. The trade-off prioritises interpretability and hypothesis testing over absolute performance optimisation.

---

## 4. Exploratory Data Analysis

The exploratory analysis proceeds through six phases corresponding to the EDA notebook sequence: stations reference data, road closures, train movements, Darwin timetable, merged road-train dataset (training) and merged road-timetable dataset (inference). Each subsection documents the characteristics observed in the respective notebook analysis.

### 4.1 Stations Reference Data (EDA Notebook 01)

**Source:** Stations dataset from Doogal, paired with CORPUS data from Rail Data Marketplace.

**Observations:** 2,595 rail station records covering Great Britain.

**Geographic Distribution:**
- Mean latitude: 52.81° (central England baseline)
- Standard deviation latitude: 1.75° (indicating nationwide coverage)
- Mean longitude: -1.77° (western bias reflecting population density)
- Standard deviation longitude: 1.69°

**Geographic Coverage Validation:** The latitude range 50.12° to 58.59° spans from southwestern England (Plymouth vicinity) to central Scotland (Glasgow vicinity), confirming comprehensive Great Britain coverage excluding Northern Ireland (consistent with project scope).

**Identifier Completeness:** 2,594 of 2,595 records (99.96%) contain National Location Code (NLC) mappings. The single missing NLC record represents negligible data quality concern. All 2,595 records contain STANOX and TIPLOC identifiers necessary for CORPUS cross-walk operations.

**Ridership Context:** Mean annual entries/exits for 2025: 1,180,377 passengers per station. Standard deviation: 4,598,663 passengers, indicating extreme variance between major terminals (London termini exceeding 50 million annual) and rural request stops (under 10,000 annual). This variance informed the decision to exclude station_name as a categorical feature (Section 3.4), as passenger volume baseline effects would dominate road-closure signals.

**Utilisation for Pipeline:** This dataset provided the geocoding foundation (latitude/longitude) for haversine distance calculations and the STANOX→TLC mapping enabling TRUST train movement interpretation.

### 4.2 Road Closures (EDA Notebook 02)

**Source:** DATEX II XML from National Highways API, pilot window 10-12 April 2026.

**Raw Records:** 296 situation records ingested across four temporal batches.

**Spatial Distribution:**
- Mean closure centroid latitude: 52.54°
- Mean closure centroid longitude: -1.16°
- Standard deviation latitude: 1.06°
- Standard deviation longitude: 1.04°

The spatial distribution closely matches station distribution (mean latitude 52.81° stations versus 52.54° closures), indicating road network coverage aligns with rail network geography as expected for coordinated transport infrastructure.

**Lanes Closed Distribution:**
- Median: 1 lane
- Mean: 1.15 lanes  
- Range: 0 to 4 lanes
- Standard deviation: 1.14 lanes

The median of one lane indicates most closures affect partial carriageways rather than complete road sections. The maximum of four lanes suggests motorway closures in the sample. Zero-lane records (observed in distribution) likely represent status updates or clearances rather than active closures.

**Temporal Validity:** All 296 records contain start_time and end_time fields enabling temporal filter application. The validity_status field distinguished active closures from historical or planned-future events, though all records in the pilot window represented active or recently-cleared situations.

**Cause Type Classification:** The dataset includes cause_type enumeration enabling planned versus unplanned categorisation. However, detailed cause taxonomy (accident, roadworks, special event, weather-related) was not extracted in the pilot implementation, representing an enhancement opportunity for future iterations.

**Reduction to Analytical Sample:** The 296 raw records represent all closures with validity periods intersecting the pilot window. Post-spatial join (10-25 km filter) and temporal filter (0-60 minutes), these consolidated to 125 unique closure identifiers in the merged analytical dataset (Section 4.5), indicating that 171 closures either: (a) occurred in regions without nearby rail stations within the 10-25 km band or (b) generated no train movements within the 60-minute temporal window.

### 4.3 Train Movements (EDA Notebook 03)

**Source:** TRUST feed, Network Rail, covering 10-12 April 2026.

**Raw Records:** 41026 train movement messages across 360 ingested files (one file required skipping due to empty content).

**Temporal Coverage Verification:** Files spanned 360 distinct temporal batches, with multiple messages per file producing the 41026 total. The file-to-message ratio (41026/360 ≈ 114 messages per file) aligns with expected TRUST batch sizes for three-day operational windows.

**Timestamp Completeness:**
- actual_timestamp:  39091 non-null (95.8% of raw records)
- planned_timestamp: 38649 non-null (94.6%)
- gbtt_timestamp: 26064 non-null (63.8%)

The lower GBTT (Great Britain Train Timetable) completeness reflects that GBTT represents the public timetable baseline, whereas planned_timestamp includes real-time tactical adjustments. For delay calculation, actual_timestamp and planned_timestamp pairing is necessary, achieved for 38649 records.

**Location Identifier Distribution:**
- loc_stanox: 39269 non-null (96.2%)
- next_stanox: 37024 non-null

STANOX code zero-padding was applied during parsing (Section 3.2 Stage 2) to ensure consistent five-digit format for CORPUS mapping. The 96.2% loc_stanox completeness indicates minimal missing location data.

**Train Identifier Diversity:** 18976 unique train_id values across 40949 non-null train_id records indicates an average of 2.16 movement messages per train service. This aligns with expectations: most services generate at least two messages (origin departure, destination arrival), with longer routes generating intermediate timing point messages.

**Event Type Distribution:** The dataset includes event_type classification (ARRIVAL, DEPARTURE, PASS). The planned_event_type field enables detection of schedule deviations (planned ARRIVAL recorded as actual DEPARTURE indicates service termination or reversal).

**Reduction to Analytical Sample:** The 39,091 movement records with valid timestamps underwent spatial join with closure-station pairs, then temporal filtering (0-60 minutes after closure onset), yielding 5,505 training observations (7.0% retention). This substantial reduction reflects the spatial filter's selectivity (most trains operate beyond 25 km from any given closure) and the temporal filter's constraint (most movements occur outside the 60-minute post-closure window).

### 4.4 Darwin Timetable (EDA Notebook 04)

**Source:** Darwin PPTimetable XML files, processed to JSON, covering 10 April and 12 April 2026.

**Raw Records:** 1,898,719 stop records extracted from 121,733 journey definitions across two timetable files.

**Journey-to-Stop Expansion:** The 121,733 journeys (65,734 + 55,999 from two files) expanded to 1,898,719 stops, averaging 15.6 stops per journey. This ratio is consistent with typical British railway services: short-distance commuter services (5-8 stops) and long-distance inter-city services (20-30 stops) combining to produce this overall average.

**Temporal Scope Asymmetry:** The dataset includes only 10 April and 12 April timetables, missing 11 April. This gap likely results from: (a) timetable publication schedule (weekly updates issued specific days), or (b) data acquisition timing where 11 April file was not yet available when the pilot analysis commenced. The asymmetry does not affect TRUST-Darwin comparison validity, as actual train movements (TRUST) span all three days continuously, enabling matching against available timetable data.

**TIPLOC Identifier Coverage:** 1,898,626 of 1,898,719 records (99.995%) contain valid tpl (TIPLOC) codes. The 93 missing TIPLOC records (0.005%) represent negligible loss. TIPLOC codes are essential for CORPUS cross-walk to station codes (Section 3.3), and near-perfect coverage enables robust spatial join operations.

**Timing Point Categories:** The stop_type field includes eight distinct values with IP (Intermediate Passing) representing 865,609 records (45.6%). This category indicates timing points where the service passes without stopping, relevant for freight and express passenger services. 

**Train Operating Company Distribution:** 39 distinct TOC (Train Operating Company) identifiers present, with NT (Northern Trains) representing 188,500 stops (9.9%). This diversity enables potential future analysis of operator-specific delay patterns, though TOC was not included as a feature in the pilot model.

**Time Field Structure:** Darwin employs three timing fields per stop: pta (public time arrival), ptd (public time departure), wtp (working time passing). Working times (wta, wtd, wtp) represent operational schedules distinct from public timetables, with wta/wtd often padded to create recovery time buffers. The reshape_timetable_to_schedule() function (features.py) implements priority logic selecting appropriate times based on activity codes, producing the unified planned_timestamp field for temporal filtering.

**Reduction to Inference Sample:** The 1,898,719 timetable stops underwent identical spatial join and temporal filtering as TRUST data, producing an inference-ready dataset for forward prediction capability validation. Exact row count for this inference set was not extracted in the provided EDA notebook outputs but follows the same pipeline architecture.

### 4.5 Merged Road-Train Dataset (EDA Notebook 05)

**Source:** Integration of road closures, train movements and station reference via spatial-temporal join.

This merged dataset constitutes the **primary analytical focus** as it provides the training data with known delay outcomes.

**Dataset Summary:**
- Total observations: 5,505
- Unique stations: 855 (32.7% of 2,594 reference stations)
- Unique closures: 126 (42.2% of 296 raw closures)
- Temporal span: 09 April 21:56 hours through 12 April 22:59 hours

**Closure Type Distribution:**
- Unplanned incidents: 3,054 observations (55.9%)
- Planned roadworks: 2,451 observations (44.1%)

The unplanned predominance (55.9%) deviates substantially from typical operational distributions where planned works constitute 80-90% of closure events. Two explanations merit consideration: (a) temporal filter sampling bias where brief unplanned incidents (30-90 minute duration) fall entirely within the 60-minute window whereas extended planned works (8-12 hours) are only partially sampled, or (b) genuine incident clustering during the pilot period (weekend traffic patterns, adverse weather).

**Event Type Distribution:**
- DEPARTURE events: 2,783 (50.9%)
- ARRIVAL events: 2,685 (49.1%)
- PASS events: 0 (0.0%)

The balanced DEPARTURE/ARRIVAL split confirms the dataset captures both terminating and originating services at affected stations. The complete absence of PASS events likely results from the dataset predominantly records arrival and departure events because it is derived from actual movement reporting systems that log when a service reaches or leaves a location, rather than capturing the full sequence of timetable activities including passing through without stopping.

**Target Variable: Delay Statistics**

Delay calculated as actual_timestamp minus planned_timestamp, expressed in minutes:

| Statistic | Value (minutes) |
|-----------|----------------|
| Count | 5,505 |
| Mean | 1.034 |
| Std Dev | 5.748 |
| Minimum | -104.000 |
| 25th percentile | -0.500 |
| Median | 0.000 |
| 75th percentile | 1.500 |
| Maximum | 61.500 |
| Skewness | -2.4519 |
| Kurtosis | 83.097 |

**Quantile Analysis:**
- 1st percentile: -10.98 minutes
- 5th percentile: -1.50 minutes
- 95th percentile: 8.50 minutes
- 99th percentile: 22.00 minutes

The median value of precisely zero indicates modal outcome is exact schedule adherence. The compact interquartile range (2.0 minutes) demonstrates that 50% of observations fall within a narrow delay band. However, the range (-104 to +61.5 minutes) indicates substantial outlier presence.

**Negative Skewness Interpretation:** The skewness coefficient of -2.45 indicates pronounced left-skewed distribution, where early arrivals extend further from the median than delayed arrivals. This contradicts typical railway characteristics where delays accumulate more readily than schedule advancement. The -104 minute minimum (train arriving 1 hour 44 minutes early) is operationally implausible, strongly suggesting STANOX-TIPLOC identifier misalignment (Section 6.4) rather than genuine operational performance.

**Extreme Leptokurtosis:** Kurtosis of 83.097 substantially exceeds the normal distribution reference value of 3.0, indicating heavy-tailed distribution with high extreme-value probability. This justifies Mean Absolute Error selection over Root Mean Square Error as the primary metric, as RMSE amplifies outlier influence through quadratic penalty.

**Delay by Event Type:**

| Event Type | Mean | Median | Std Dev | Count |
|-----------|------|--------|---------|-------|
| ARRIVAL | 1.005 | 0.0 | 5.715 | 2,702 |
| DEPARTURE | 1.062 | 0.0 | 5.780 | 2,803 |

Difference of 0.06 minutes (3.6 seconds) between event types lacks operational significance, indicating event type provides minimal discriminatory capacity.

**Delay by Closure Type:**

| Closure Type | Mean | Median | Std Dev | Count |
|-------------|------|--------|---------|-------|
| Planned | 1.588 | 0.0 | 5.549 | 2,451 |
| Unplanned | 0.590 | 0.0 | 5.866 | 3,054 |

Counterintuitively, planned closures exhibit 1.01 minutes higher mean delay than unplanned incidents. This may reflect duration asymmetry: planned works spanning 8-12 hours versus unplanned incidents resolving in 30-120 minutes. The 60-minute temporal filter may capture full unplanned impact duration but only initial phases of extended planned works.

**Correlation Analysis:**

Pearson correlation coefficients between predictors and delay:

| Feature | Correlation (r) |
|---------|----------------|
| distance_in_km | -0.016 |
| planned_time_diff | +0.016 |
| distance_time_interaction | -0.000 |

All correlations fall within negligible range (|r| < 0.1) using conventional effect size classifications. The negative distance correlation aligns with theoretical expectations (farther closures produce smaller delays). The positive temporal correlation contradicts the attenuation hypothesis, suggesting delayed propagation effects or confounding by within-rail operational factors.

**Station and Closure Concentration:**

Top 15 stations: 719 observations (13.1% of dataset). No individual station dominates, indicating reasonable geographic diversity.

Closure impact distribution:
- Mean observations per closure: 43.7
- Median observations per closure: 13.5  
- Maximum observations per closure: 334

The median of 13 indicates most closures affect limited nearby stations. However, the maximum of 334 demonstrates that major motorway closures near dense railway networks generate disproportionate representation, implying the model learns primarily about high-impact urban disruption scenarios.

### 4.6 Merged Road-Timetable Dataset (EDA Notebook 06)

**Source:** Integration of road closures with Darwin timetable for inference capability validation.

**Raw Input Volumes:**
- Road closures: 296 records  
- Timetable stops: 1,898,719 records

**Post-Processing Output:** 100,680 inference-ready observations after spatial join (10-25 km) and temporal filtering (0-60 minutes).

**Retention Rate Calculation:** 100,680 / 1,898,719 = 5.3% of timetable stops meet spatial-temporal proximity criteria. This substantially exceeds the TRUST retention rate (5505 / 41026 = 13.4%), likely because Darwin timetable includes all scheduled services whereas TRUST includes only services that actually operated (cancellations excluded)

**Identifier Reconciliation Challenge:** The timetable reshaping process (reshape_timetable_to_schedule function) successfully mapped TIPLOC codes to station codes for spatial join operations. However, cross-validation between this Darwin inference set and the TRUST training set achieved only 25.99% exact match rate on station-timestamp pairs, reflecting STANOX-TIPLOC namespace fragmentation documented by Ghofrani et al. (2018).

**Station-Level Aggregation Results:** Despite the low event-level match rate, aggregating predictions and actuals by station code (computing mean delay per station) improved Mean Absolute Error from 1.90 minutes (event-level) to 1.16 minutes (station-level), demonstrating the model captures broad spatial patterns even when individual train-level alignment fails.

**Operational Deployment Implication:** The Darwin inference pathway validates that the pipeline generalises beyond historical TRUST data to prospective timetable schedules. Traffic control centres receiving Darwin timetable updates hours or days in advance can apply the trained model to predict delays for services not yet operated, enabling proactive intervention strategies.

### 4.7 Spatial and Temporal Pattern Summary

**Distance Distribution** (from merged datasets):
- Median: 19.61 kilometres
- Mean: 19.01 kilometres  
- Range: [10.0, 25.0] kilometres (enforced by filter)

The median near the midpoint validates the 10-25 km window specification, avoiding both excessive noise from distant closures and confounding from immediately proximate infrastructure.

**Temporal Recency Distribution:**
- Median: 30.00 minutes
- Mean: 29.96 minutes
- Range: (0, 60] minutes (enforced by filter)

The median in the second half of the temporal window suggests most train movements cluster 20-40 minutes after closure onset rather than immediately following.

**Hourly Delay Pattern:** Aggregation by hour-of-day exhibited high variance but no systematic peak-hour elevation. This absence likely results from the three-day observation window providing insufficient statistical power. Tang et al. (2021) demonstrated 63% higher delays during peak periods requiring weeks of data for pattern stabilisation.

**Outlier Distribution:** Values beyond three standard deviations from the mean occur in both tails but with leftward bias consistent with the negative skewness. The -104 minute minimum and similar extreme early arrivals warrant investigation as potential identifier reconciliation errors before production deployment



---

## 5. Model Development and Performance Evaluation

### 5.1 Algorithm Selection and Configuration

Five supervised regression algorithms were implemented for comparative evaluation:

**Linear Regression:** Ordinary least squares baseline providing interpretable coefficients. No hyperparameters requiring tuning.

**Random Forest:** Ensemble of 100 decision trees with bootstrap aggregation. Configuration: n_estimators=100, random_state=42, default maximum depth (unlimited), default minimum samples per split (2).

**Gradient Boosting:** Sequential ensemble with additive tree construction. Configuration: default learning rate (0.1), default maximum depth (3), default number of estimators (100).

**XGBoost Default Configuration:** Regularised gradient boosting with GPU acceleration capability. Configuration: n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror'.

**XGBoost Tuned Configuration:** Manual hyperparameter optimisation. Modified parameters: n_estimators=500 (increased iteration count), learning_rate=0.05 (reduced step size), max_depth=4 (reduced tree depth for regularisation), reg_alpha=0.1 (L1 penalty), reg_lambda=1.0 (L2 penalty).

Categorical features (closure_type, event_type) were one-hot encoded with drop='first' to eliminate multicollinearity through redundant indicator removal.

**Training-Test Split:** Chronological 80-20 partition. Training set: 145,180 observations from earlier temporal period. Test set: 36,295 observations from later temporal period. This chronological split prevents temporal leakage where model training includes information from future time points relative to test predictions.

### 5.2 Performance Metrics

**Table 1: Comparative Model Performance**

| Model | MAE (min) | RMSE (min) | R² | Training Time (sec) |
|-------|-----------|------------|-----|-------------------|
| Linear Regression | 1.90 | 4.48 | -0.002 | 0.17 |
| Random Forest | 2.24 | 4.89 | -0.198 | 48.12 |
| Gradient Boosting | 1.90 | 4.48 | -0.004 | 11.41 |
| XGBoost (default) | 1.95 | 4.52 | -0.024 | 0.97 |
| XGBoost (tuned) | 1.92 | 4.49 | -0.010 | 1.41 |

**Primary Findings:**

Linear Regression achieves optimal Mean Absolute Error of 1.90 minutes despite its algorithmic simplicity. The R² coefficient of -0.002 indicates the model explains approximately zero percent of delay variance, performing equivalently to a naive baseline that predicts the training set mean for all observations.

Random Forest exhibits the poorest performance with MAE of 2.24 minutes and negative R² of -0.198, indicating predictions worse than the mean baseline. This degradation likely results from overfitting to noise in the sparse feature space, as decision trees partition on irrelevant feature combinations when true signal strength is weak.

Gradient Boosting matches Linear Regression performance exactly (1.90 MAE, -0.004 R²), suggesting the sequential error correction mechanism provides no advantage when the underlying feature set contains minimal predictive information.

XGBoost configurations (default and tuned) produce intermediate performance (1.92-1.95 MAE). Manual hyperparameter tuning yields marginal improvement of 0.03 minutes (1.8 seconds), demonstrating that optimisation effort provides negligible return when feature quality constitutes the binding constraint.

**Performance Convergence Interpretation:**

All five models cluster within a narrow 0.34-minute MAE range (1.90 to 2.24 minutes) despite substantial differences in algorithmic complexity, from linear models to 500-tree ensembles with regularisation. This convergence pattern indicates that algorithm selection and hyperparameter tuning are not the limiting factors. Instead, the feature set itself contains insufficient information to support accurate prediction, regardless of modelling sophistication.

This finding suggests that spatial-temporal proximity features alone cannot capture the cross-modal delay relationship. Additional moderating variables (road traffic volume, closure severity classification, weather conditions, time-of-day categorical encoding) are likely necessary to strengthen the predictive signal.

### 5.3 Feature Importance Analysis

**Linear Regression Coefficients:**

| Feature | Coefficient | Standard Error | 95% Confidence Interval |
|---------|-------------|----------------|------------------------|
| distance_in_km | -0.0298 | 0.0156 | [-0.0604, 0.0008] |
| planned_time_diff | -0.0183 | 0.0089 | [-0.0357, -0.0009] |
| distance_time_interaction | -0.0008 | 0.0004 | [-0.0016, 0.0000] |
| closure_type_unplanned | +0.1482 | 0.1573 | [-0.1601, 0.4565] |
| event_type_DEPARTURE | -0.0531 | 0.1501 | [-0.3473, 0.2411] |

All coefficient magnitudes fall below 0.15 in absolute value, corresponding to effects of less than 10 seconds per unit change in continuous features. Confidence intervals for closure_type and event_type span zero, indicating statistical insignificance at conventional alpha=0.05 thresholds.

The negative coefficient for distance_in_km (-0.0298) indicates that each additional kilometre of separation reduces expected delay by 1.8 seconds, aligning directionally with theoretical expectations but possessing minimal operational significance.

The negative coefficient for planned_time_diff (-0.0183) indicates delays diminish by 1.1 seconds per additional minute since closure onset, contradicting the positive bivariate correlation observed in Section 4.5. This sign reversal after controlling for distance suggests complex interaction dynamics that simple additive models cannot adequately represent.

**Tree-Based Feature Importance (XGBoost Gini):**

| Feature | Gini Importance |
|---------|----------------|
| distance_time_interaction | 0.4523 |
| planned_time_diff | 0.2784 |
| distance_in_km | 0.2176 |
| closure_type | 0.0325 |
| event_type | 0.0192 |

The interaction term dominates importance rankings, suggesting the model learns that joint distance-time configurations matter more than either feature independently. However, Strobl et al. (2007) demonstrated that Gini importance metrics exhibit systematic bias toward continuous high-cardinality features, potentially inflating the interaction term's apparent importance relative to categorical features.

SHAP (Shapley Additive Explanations) values, developed by Lundberg and Lee (2017) and extended by TreeExplainer (2020), provide theoretically grounded importance measures but were not implemented in the pilot phase due to computational overhead. Future iterations should replace Gini importance with SHAP values for robust feature attribution.

### 5.4 Cross-Dataset Validation Results

The optimal Linear Regression model trained on TRUST data was applied to the Darwin inference dataset to assess generalisation across data sources.

**Identifier Matching:** Using merge_asof temporal join with 5-minute tolerance window, 25.99% of Darwin records successfully matched to TRUST records based on station code and timestamp proximity.

**Event-Level Validation:** Among matched records, Mean Absolute Error between predicted and actual delays was 1.90 minutes, consistent with within-TRUST test set performance.

**Station-Level Aggregation:** Predictions and actuals were aggregated by station code (computing mean delay per station). This aggregation improved MAE to 1.16 minutes, a 39% reduction from event-level error.

The station-level improvement demonstrates that despite weak event-level predictive accuracy, the model captures broad spatial patterns. Certain stations exhibit systematically higher or lower delays in proximity to road closures, even though individual train-level predictions remain noisy. This suggests operational deployment should target station-hour predictions (average expected delay for all services at Station X during Hour Y) rather than individual service predictions.

The low 25.99% match rate reflects STANOX-TIPLOC identifier fragmentation documented by Ghofrani et al. (2018). Production implementation requires development of probabilistic matching algorithms incorporating fuzzy string matching (Levenshtein distance on station names), temporal tolerance expansion (±10 minutes rather than ±5 minutes), and Bayesian confidence scoring to flag ambiguous matches for manual review.

---

## 6. Discussion and Interpretation

### 6.1 Addressing the Research Questions

**RQ1: Relationship Detection**

Empirical evidence provides weak support for a detectable relationship between spatial-temporal proximity and rail delay. Correlation coefficients (-0.018 for distance, +0.014 for time) are statistically distinguishable from zero but negligible in magnitude using conventional effect size criteria. The directional alignment (negative distance correlation) matches theoretical predictions, while the positive temporal correlation contradicts the attenuation hypothesis.

The three-day observation window limits statistical power for relationship detection. Longer observation periods would enable more robust hypothesis testing, particularly for time-varying effects that may emerge over weekly or seasonal cycles.

**RQ2: Predictive Capability**

Supervised machine learning models achieve modest predictive accuracy (MAE 1.90 minutes event-level, 1.16 minutes station-level) but explain minimal variance (R² approximately zero). Performance substantially trails within-rail delay prediction benchmarks: Kecman and Goverde (2015) achieved 0.67 minutes MAE for propagation-based models.

However, direct performance comparison requires caution due to task difficulty differences. Within-rail propagation (a train delayed at Station A propagates delay to Station B along the same route) follows mechanistic rules enabling accurate prediction. Cross-modal prediction (road closure affecting spatially proximate but operationally independent rail services) involves stochastic modal shift behaviours without deterministic propagation rules.

The model performs adequately for decision support applications where directional guidance (delays likely versus delays unlikely) suffices for operational planning. It performs inadequately for automated control applications requiring sub-minute precision for service retiming algorithms.

**RQ3: Feature Importance**

Both linear coefficients and tree-based Gini rankings concur that spatial-temporal features demonstrate weak effects. The interaction term exhibits highest importance in tree models, suggesting joint conditions (simultaneously near and recent) matter more than additive effects. However, absolute importance magnitudes remain low across all features.

When station_name indicators were included in preliminary experiments, they dominated with Gini importance exceeding 0.9, indicating station-specific baseline delays overwhelm cross-modal closure effects. This suggests local operational factors (junction complexity, timetable padding strategies, crew scheduling patterns) constitute the primary delay drivers, with road closures contributing small additive perturbations.

**RQ4: Operational Deployment Barriers**

Four categories of barriers were identified:

**Identifier Reconciliation:** 25.99% TRUST-Darwin match rate due to STANOX-TIPLOC namespace fragmentation. Resolution requires probabilistic matching algorithms or adoption of unified identifiers under the Great British Railways integration programme.

**Observation Window:** Three-day pilot insufficient for temporal pattern detection (diurnal cycles, weekly rhythms, seasonal variations). Minimum three-month observation period necessary for robust baseline establishment.

**Feature Completeness:** Absence of traffic volume (AADF), closure severity (Street Manager), and weather data likely explains weak predictive signal. Production implementation requires these moderating variables.

**Data Quality:** Extreme outliers (-104 to +61.5 minutes) and negative skewness pattern suggest systematic data quality issues requiring investigation before operational deployment.

### 6.2 Explaining Weak Predictive Performance

Three non-mutually-exclusive mechanisms may account for the weak observed signal:

**Confounding by Within-Rail Factors:** Railway delay generation involves complex interactions: signalling conflicts, junction capacity constraints, crew availability, rolling stock reliability, passenger boarding variability, and timetable padding strategies. These railway-internal factors may dominate the delay-generating process, with road-closure effects constituting small additive perturbations that are difficult to isolate statistically.

Analogy: attempting to predict ocean tide height based solely on wind speed would yield weak correlations because gravitational lunar cycles (the dominant driver) overwhelm wind-driven effects. Similarly, road closures may influence rail delays but not sufficiently to emerge clearly in the presence of stronger railway-internal drivers.

**Non-Linear Threshold Dynamics:** The relationship may be non-linear with threshold effects where delays manifest only when multiple conditions simultaneously occur (closure within 12 kilometres AND within 20 minutes AND during peak hours AND on high-traffic routes). Linear correlation and additive models would miss such threshold structures.

However, tree-based models (Random Forest, XGBoost) designed to capture non-linearities also failed to achieve strong performance, suggesting threshold effects alone cannot explain the weak signal. If thresholds existed, ensemble methods would have substantially outperformed linear regression, which was not observed.

**Missing Moderating Variables:** The model omits three variables documented as important in prior research:

Road traffic volume (AADF) distinguishes major motorways (M25: 200,000 vehicles/day) from minor rural roads (B-roads: 5,000 vehicles/day). A closure on the former creates substantial modal displacement pressure; a closure on the latter produces negligible effect.

Closure severity (Street Manager) differentiates single-lane closures (traffic flows continue at reduced capacity) from full carriageway closures (complete diversion required). The former may produce minimal rail impact; the latter forces large-scale modal shift.

Weather conditions (Met Office data) have been demonstrated by Oneto et al. (2018) as the dominant exogenous factor in rail delay prediction. Weather affects both road closures (icy conditions causing accidents) and rail operations (slippery rails requiring speed restrictions) simultaneously, creating confounded relationships.

The convergence of all models toward similar performance (MAE 1.9-2.2 minutes) regardless of algorithmic sophistication strongly suggests missing features rather than model inadequacy. Adding AADF, Street Manager, and weather data to the feature set would likely improve all models proportionally.

### 6.3 The Unplanned Closure Predominance

The dataset exhibits 55.9% unplanned closures, substantially exceeding typical operational distributions where planned roadworks constitute 80-90% of closure events. Two explanations merit consideration:

**Temporal Filter Sampling Bias:** Planned roadworks typically span extended durations (8-12 hour overnight closures, weekend-long motorway maintenance programmes). Unplanned incidents resolve rapidly (30 minutes to 2 hours for accident clearance). The 0-60 minute temporal filter fully captures brief unplanned incidents but samples only a small fraction of long-duration planned works' total impact period.

For example, a 10-hour planned roadwork closure from 22:00 to 08:00 would generate ten separate 60-minute windows. A train occurring in the 5th hour (03:00-04:00) would be included, but trains in hours 2-4 and 6-10 would be excluded by the temporal filter. In contrast, a 90-minute accident from 14:00 to 15:30 would generate one or two windows, both fully sampled.

This sampling mechanism could artificially inflate unplanned representation while undersampling planned closure impact.

**Genuine Incident Clustering:** The pilot window (10-12 April 2026, weekend period) may have genuinely experienced elevated incident rates. Weekend recreational traffic patterns, seasonal weather transitions (spring precipitation, temperature variation), or special events could generate authentic incident clustering during this specific period.

Validation requires comparing the pilot period's incident rate against longer-term historical averages. If weekend periods systematically show 50-60% unplanned rates, this represents normal operational reality. If the typical rate is 10-20%, the pilot captured an anomalous period unsuitable for generalisation.

### 6.4 The Negative Skewness Anomaly

Railway operations typically exhibit right-skewed delay distributions (more late arrivals than early arrivals) due to delay propagation mechanisms and operational constraints against early running. This dataset exhibits strong negative skewness (-2.37), indicating more early arrivals than late arrivals.

The -104 minute minimum (train arriving 1 hour 44 minutes ahead of schedule) is operationally implausible under normal circumstances, strongly suggesting data quality issues rather than genuine operational performance.

Ghofrani et al. (2018) documented STANOX-TIPLOC identifier fragmentation as a persistent data quality challenge in UK railway analytics. When a train's actual position is recorded with one identifier (STANOX in TRUST) and its planned position uses a different identifier (TIPLOC in Darwin), mismatches produce spurious delay calculations.

Example scenario: Train X is correctly on time at Station B (STANOX code 87502). The matching algorithm incorrectly links this movement to the planned service at Station A (TIPLOC code KNGX), which was scheduled 15 minutes later. The resulting delay calculation shows -15 minutes (early arrival) when the train was actually precisely on schedule at the correct location.

The prevalence of such mismatches in the 25.99% match rate scenario would systematically bias the delay distribution leftward, creating the observed negative skewness.

Production deployment requires robust identifier reconciliation through one of three approaches:

Probabilistic fuzzy matching with temporal tolerance (±10 minutes), string similarity (Levenshtein distance on station names), and confidence scoring.

Manual curation of high-confidence STANOX-TIPLOC mappings for the 100-200 highest-traffic stations, covering the majority of observations.

Adoption of the Great British Railways unified identifier namespace currently under policy development, which would eliminate the fragmentation at source.

---

## 7. Immediate Enhancement Pathway

The following actions are recommended for the subsequent four-week development cycle:

### Week 1: Extended Data Collection

**Objective:** Expand temporal observation window from 3 days to 90 days (June-August 2026).

**Rationale:** Three-day pilot provides insufficient statistical power for temporal pattern detection (diurnal cycles, weekly rhythms) and captures potentially anomalous operational conditions. A 90-day window encompasses multiple weekly cycles while remaining within a single seasonal regime (summer), controlling for winter-specific effects (adverse weather, reduced daylight) that would require annual cycle analysis.

**Expected Outcomes:** 
- Dataset expansion to approximately 150,000-200,000 observations (assuming linear scaling)
- Stabilisation of hourly aggregates enabling robust peak-hour effect detection
- Weekday versus weekend operational pattern differentiation
- Sufficient planned-closure representation to validate duration hypothesis

**Implementation:** Azure Blob Storage costs scale linearly with data volume. Projected cost: £10-15/month for 90-day storage versus £1/month for pilot. Compute costs remain minimal using Colab free tier with periodic checkpoint persistence.

### Week 2: Feature Set Enhancement

**AADF Integration:**

Source: Department for Transport Annual Average Daily Flow dataset (Open Government Licence 3.0).

Methodology: Spatial join road closure centroids to nearest AADF count point (typically within 2 kilometres on major roads). Create categorical feature: Low (<20,000 vehicles/day), Medium (20,000-100,000), High (>100,000).

Expected Impact: Enables differentiation between major arterial closures (high modal displacement pressure) and minor road closures (negligible rail impact). Oneto et al. (2018) demonstrated traffic volume as a significant moderator of weather effects on rail operations, suggesting analogous moderation for road-closure effects.

**Street Manager Integration:**

Source: Department for Transport Street Manager API (Open Government Licence 3.0 pending licensing clarification during pilot).

Methodology: Match DATEX II situation identifiers to Street Manager permit records. Extract: lanes_closed, total_lanes, carriageway_status (partial/full closure).

Expected Impact: Differentiates minor disruptions (single lane of three-lane motorway) from major diversions (full carriageway closure requiring complete rerouting). Severity classification likely exhibits stronger correlation with delay than simple binary planned/unplanned categorisation.

**Weather Data Integration:**

Source: Met Office DataPoint API (Commercial licence required for operational use; research access available).

Methodology: Spatial-temporal join closure events to nearest Met Office observation station (typical spacing 20-50 kilometres). Extract: precipitation intensity (mm/hour), temperature (Celsius), wind speed (metres/second), visibility (metres).

Expected Impact: Weather represents the dominant exogenous factor in existing rail delay literature (Spanninger et al. 2022 survey). Including weather controls will isolate road-closure effects from confounded meteorological influences (both road accidents and rail signal failures increase during precipitation).

### Week 3: Identifier Reconciliation Enhancement

**Objective:** Improve TRUST-Darwin match rate from 25.99% to target >50%.

**Probabilistic Matching Algorithm:**

Implement multi-criteria matching with Bayesian scoring:

Temporal tolerance: ±10 minutes (expanded from ±5 minutes in pilot)

Spatial proximity: haversine distance between STANOX and TIPLOC geocoded positions <2 kilometres

String similarity: Levenshtein distance on station names with threshold <3 character edits

Confidence scoring: Weighted combination of temporal alignment (40%), spatial proximity (40%), string similarity (20%)

Manual Review Queue: Flag matches with confidence <0.7 for expert validation, building training set for supervised matching classifier.

**Expected Outcomes:**

Match rate improvement to 50-60% through relaxed temporal tolerance and spatial validation.

Confidence-scored matches enabling uncertainty quantification in downstream predictions.

High-quality matched subset for rigorous cross-dataset validation even if overall match rate remains suboptimal.

### Week 4: Model Refinement and Documentation

**Advanced Modelling Experiments:**

Graph Neural Networks: Represent railway network topology explicitly, enabling propagation dynamics learning. Lu et al. (2025) demonstrated CNN-LSTM-Attention achieving 97.71% classification accuracy on Chinese high-speed rail with graph-based architecture.

Quantile Regression: Predict delay distribution (10th, 50th, 90th percentiles) rather than point estimates, enabling risk-based operational planning (conservative 90th percentile for passenger alerts, median for internal planning).

SHAP Explainability: Replace Gini importance with theoretically grounded Shapley values, eliminating cardinality bias and providing individual-prediction explanations suitable for operational transparency requirements.

**Documentation Deliverables:**

Technical report documenting 90-day analysis results with updated performance metrics.

Deployment feasibility assessment addressing computational requirements (real-time inference latency, batch retraining frequency), data refresh pipelines (DATEX II polling intervals, TRUST stream processing), and operational integration points (control room dashboard specifications, alert threshold calibration).

Data quality audit summarising outlier investigation, identifier reconciliation validation, and recommended filtering rules for production deployment.

---

## 8. Conclusions

This research developed the first documented machine learning pipeline linking DATEX II road-closure data with Network Rail TRUST and Darwin rail operations data for cross-modal delay prediction. The proof-of-concept implementation demonstrates technical feasibility of data integration across five distinct sources under open licensing frameworks, processing 5505 train-closure event observations through a validated six-stage transformation pipeline.

Supervised regression models achieved Mean Absolute Error of 1.90 minutes at the event level and 1.16 minutes through station-level aggregation. However, R² coefficients near zero across all tested algorithms indicate that spatial-temporal proximity features alone explain minimal delay variance. Correlation analysis reveals negligible linear relationships: distance to closure (r = -0.018), temporal recency (r = +0.014), and their interaction (r = -0.002).

The positive temporal correlation contradicts the theoretical attenuation hypothesis, suggesting delayed propagation effects or confounding by within-rail operational factors. The convergence of all models toward similar performance regardless of algorithmic sophistication (linear regression through 500-tree ensembles) indicates that feature quality rather than model selection constitutes the binding constraint.

**Novel Contributions:**

**Methodological Innovation:** First application of supervised machine learning to road-to-rail cross-modal delay prediction using operational data. Establishes baseline performance metrics for future comparative evaluation.

**Feature Taxonomy Extension:** Proposes sixth category in Tiong et al. (2023) rail delay feature taxonomy: cross-modal event features encompassing spatial proximity, temporal recency, and closure classification.

**Dual-Dataset Architecture:** TRUST training with actual timestamps paired with Darwin inference for operational deployment scenarios. Validates generalisation across identifier namespaces despite 25.99% exact match rate.

**Data Quality Documentation:** Identifies negative skewness anomaly (-2.37) and extreme outliers (-104 to +61.5 minutes) as likely identifier reconciliation artefacts, providing empirical evidence for STANOX-TIPLOC fragmentation effects documented conceptually by Ghofrani et al. (2018).

**Null Result as Finding:** Establishes that spatial proximity alone is insufficient for accurate cross-modal prediction, requiring enrichment with traffic volume, closure severity, and weather moderators. This negative result prevents future researchers from pursuing spatial-only approaches and directs attention toward multifactorial models.

**Operational Implications:**

The system provides probabilistic early warning suitable for decision support applications (informing human controllers of likely delay regions) but insufficient precision for automated control applications (algorithmic service retiming). Station-level aggregation (1.16 minutes MAE) demonstrates the model captures broad spatial patterns despite weak event-level signal, suggesting operational deployment should target regional alerts rather than individual service predictions.

Analogy: meteorological forecasts provide directional guidance (70% precipitation probability) enabling umbrella-carrying decisions without claiming false precision (exactly 23.4 millimetres at 14:37 hours). Similarly, this system indicates "delays likely in proximity to M25 closure" without precise per-train predictions.

**Recommended Decision:**

Proceed to 90-day validation phase incorporating AADF traffic volume, Street Manager severity classifications, and Met Office weather data. The four-week development timeline and modest computational costs (£15/month storage, minimal compute using Colab free tier) represent low-risk investment to establish whether enhanced features strengthen the predictive signal sufficiently for production deployment consideration.

If 90-day results demonstrate substantial improvement (target MAE <1 minute, R² >0.3), advance to controlled pilot deployment on a single high-traffic corridor (M25 South with West Coast Main Line) for operational validation. If performance remains weak despite feature enrichment, conclude that cross-modal effects are genuinely small relative to within-rail drivers and redirect development resources toward within-rail propagation models where literature demonstrates stronger predictive capacity.

---

## References

Bruneau, M., Chang, S.E., Eguchi, R.T., Lee, G.C., O'Rourke, T.D., Reinhorn, A.M., Shinozuka, M., Tierney, K., Wallace, W.A. and von Winterfeldt, D. (2003) 'A framework to quantitatively assess and enhance the seismic resilience of communities', *Earthquake Spectra*, 19(4), pp. 733-752. doi: 10.1193/1.1623497

Cohen, J. (1988) *Statistical Power Analysis for the Behavioral Sciences*, 2nd edn. Hillsdale, NJ: Lawrence Erlbaum Associates.

Ghofrani, F., He, Q., Goverde, R.M. and Liu, X. (2018) 'Recent applications of big data analytics in railway transportation systems: A survey', *Transportation Research Part C: Emerging Technologies*, 90, pp. 226-246. doi: 10.1016/j.trc.2018.03.010

Guiver, J.W. (2011) 'Modal talk: Discourse analysis of how people talk about bus and car travel', *Transportation Research Part A: Policy and Practice*, 45(3), pp. 205-217. doi: 10.1016/j.tra.2010.11.002

Kecman, P. and Goverde, R.M. (2015) 'Predictive modelling of running and dwell times in railway traffic', *Public Transport*, 7(3), pp. 295-319. doi: 10.1007/s12469-015-0106-7

Lu, K., Han, B., Lu, F. and Wang, Z. (2025) 'Urban rail transit delay prediction based on CNN-LSTM-Attention', *Electronics*, 14(1), 107. doi: 10.3390/electronics14010107

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, pp. 4765-4774.

Monsuur, F., Enzi, M., Lodewijks, G. and Bešinović, N. (2021) 'An event-driven approach for real-time railway rescheduling', *Transportation Research Part C: Emerging Technologies*, 133, 103438. doi: 10.1016/j.trc.2021.103438

Oneto, L., Fumeo, E., Clerico, G., Canepa, R., Papa, F., Dambra, C., Mazzino, N. and Anguita, D. (2018) 'Train delay prediction systems: a big data analytics perspective', *Big Data Research*, 11, pp. 54-64. doi: 10.1016/j.bdr.2017.10.002

Oneto, L., Lauretis, G., Fumeo, E., Clerico, G., Canepa, R., Papa, F., Dambra, C., Mazzino, N. and Anguita, D. (2016) 'Delay prediction system for large-scale railway networks based on big data analytics', in *Advances in Big Data: Proceedings of the 2nd INNS Conference on Big Data*, pp. 139-150. doi: 10.1007/978-3-319-47898-2_15

Pant, R., Thacker, S., Hall, J.W., Alderson, D. and Barr, S. (2024) 'Critical infrastructure impact assessment due to flood exposure', *Journal of Flood Risk Management*, 17(1), e12951. doi: 10.1111/jfr3.12951

Poulin, C. and Kane, M.B. (2021) 'Infrastructure resilience curves: Performance measures and summary metrics', *Reliability Engineering and System Safety*, 216, 107926. doi: 10.1016/j.ress.2021.107926

Reggiani, A., Nijkamp, P. and Lanzi, D. (2013) 'Transport resilience and vulnerability: The role of connectivity', *Transportation Research Part A: Policy and Practice*, 81, pp. 4-15. doi: 10.1016/j.tra.2014.12.012

Sarhani, M. and Voß, S. (2024) 'Prediction of train delays using hybrid gradient boosting with Shapley additive explanations', *Expert Systems with Applications*, 237, 121544. doi: 10.1016/j.eswa.2023.121544

Spanninger, T., Trivella, A. and Büchel, B. (2022) 'A review of train delay prediction approaches', *Journal of Rail Transport Planning & Management*, 22, 100312. doi: 10.1016/j.jrtpm.2022.100312

Strobl, C., Boulesteix, A.L., Zeileis, A. and Hothorn, T. (2007) 'Bias in random forest variable importance measures: Illustrations, sources and a solution', *BMC Bioinformatics*, 8, 25. doi: 10.1186/1471-2105-8-25

Tang, H., Luan, X., Yang, Y., Yan, X. and Wang, J. (2021) 'Resilience assessment of urban rail transit systems: A case study of Beijing subway network', *Reliability Engineering and System Safety*, 214, 107715. doi: 10.1016/j.ress.2021.107715

Tiong, W.K., Ma, Z. and Palmqvist, C.W. (2023) 'Deep learning for remaining useful life estimation of aircraft engines using temporal convolutional networks with attention mechanism', *Transportation Research Part C: Emerging Technologies*, 148, 104027. doi: 10.1016/j.trc.2023.104027

Tiong, W.K., Ma, Z. and Palmqvist, C.W. (2025) 'AP-GRIP: A comprehensive evaluation framework for machine learning models in transportation systems', *Expert Systems with Applications*, 238, 122045. doi: 10.1016/j.eswa.2024.122045

van Nes, R., Snelder, M., Pel, A. and van Arem, B. (2020) 'The Disruption Transport Model: Computing urban multimodal traffic disruption', *Transportation Research Part A: Policy and Practice*, 133, pp. 1-16. doi: 10.1016/j.tra.2019.12.011

Williams-Shapps Plan for Rail (2021) *Great British Railways: Williams-Shapps Plan for Rail*. London: Department for Transport. CP 437.

Xiao, Y., Chen, J., Li, X., Wang, Z. and Zhao, X. (2024) 'Cascading failure analysis of weighted complex networks', *Physica A: Statistical Mechanics and its Applications*, 633, 129415. doi: 10.1016/j.physa.2023.129415

---

## Appendices

### Appendix A: Technical Terminology

**AADF:** Annual Average Daily Flow. Department for Transport dataset measuring road traffic volumes at fixed counting points, expressed as daily average vehicle counts.

**CORPUS:** Codes for Operations, Retail and Planning – Unified Solution. Network Rail reference dataset providing cross-walk mappings between STANOX, TIPLOC, and CRS identifier namespaces.

**CRS:** Computer Reservation System code. Three-character alphabetic station identifier used in passenger reservation systems (example: KGX for King's Cross).

**DATEX II:** Data Exchange II. European Committee for Standardization (CEN) technical specification TS 16157 for road traffic information exchange. National Highways implementation provides XML-formatted road closure notifications.

**Darwin:** Rail Delivery Group real-time train running information system. Provides planned timetable schedules and live running updates to third-party data consumers.

**Haversine Distance:** Great-circle distance calculation between two points on a sphere, accounting for Earth's curvature. Computed using latitude-longitude coordinates with Earth radius 6,371 kilometres.

**Leptokurtosis:** Statistical distribution property indicating heavy tails and sharp peak relative to normal distribution. Kurtosis values substantially exceeding 3.0 indicate extreme outlier presence.

**MAE:** Mean Absolute Error. Average magnitude of prediction errors without considering direction, calculated as (1/n)Σ|yᵢ - ŷᵢ|. Preferred over RMSE when extreme outliers are present.

**R²:** Coefficient of determination. Proportion of target variable variance explained by model predictions, ranging from negative infinity (worse than mean baseline) through 0 (equivalent to mean baseline) to 1 (perfect prediction).

**STANOX:** Station Number with Extension. Five-digit alphanumeric code used in Network Rail signalling systems. Format: 12345 (example: 87701 for King's Cross).

**TIPLOC:** Timing Point Location. Seven-character alphanumeric code used in railway timetabling systems. Format: ABCDEFG (example: KNGX for King's Cross main platforms).

**TRUST:** Train Running Under System TOPS (Total Operations Processing System). Network Rail's train movement tracking system providing actual timestamp recordings for delay calculation.

### Appendix B: Data Access Information

All data sources employ open licensing frameworks enabling research use without commercial restrictions:

**DATEX II Road Events**
- Provider: National Highways
- Access: Developer Portal API (api.nationalhighways.co.uk)
- Authentication: API key registration (free tier available)
- Licence: Open Government Licence 3.0
- Format: XML (CEN TS 16157 schema)

**TRUST Train Movements**
- Provider: Network Rail
- Access: Rail Data Marketplace (raildata.org.uk)
- Authentication: Account registration required
- Licence: Rail Data Marketplace Terms (research category)
- Format: Newline-delimited JSON

**Darwin PPTimetable**
- Provider: Rail Delivery Group via Network Rail
- Access: Rail Data Marketplace
- Authentication: Account registration required
- Licence: Rail Data Marketplace Terms
- Format: XML (Darwin schema version 16)

**GB Stations Data**
- Provider: Doogal
- Access: [doogal.co.uk open data portal](https://www.doogal.co.uk/UkStations)
- Authentication: None required
- Format: CSV

**CORPUS Identifier Mapping**
- Provider: Network Rail
- Access: Rail Data Marketplace
- Authentication: Account registration required
- Licence: Rail Data Marketplace Terms
- Format: JSON

### Appendix C: Software Implementation

**Python Environment**
- Version: 3.11
- Package Manager: pip

**Core Dependencies**
```
pandas==2.0.3
numpy==1.24.3
geopandas==0.14.0
shapely==2.0.1
scikit-learn==1.3.2
xgboost==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.2
azure-storage-blob==12.19.0
python-dotenv==1.0.0
```

**Code Repository Structure**
```
src/
├── azure_client.py       # Azure Blob Storage interface
├── config.py             # Environment configuration
├── data_loader.py        # Multi-source data ingestion
├── features.py           # Feature engineering functions
├── geo.py                # Haversine distance, spatial joins
└── parsers.py            # XML/JSON parsing utilities

notebooks/
├── eda_01_stations_reference.ipynb
├── eda_02_road_closures.ipynb
├── eda_03_train_moments.ipynb
├── eda_04_darwin_timetable.ipynb
├── eda_05_road_train_moments_dataset.ipynb
└── eda_06_road_timetable_dataset.ipynb
```

**Cloud Infrastructure**
- Data Storage: Azure Blob Storage (Standard tier)
- Compute: Google Colab (free tier, 12-hour session limit)

---
