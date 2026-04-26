# Flight Delay Prediction

Predicts whether a U.S. domestic flight will be delayed using the
[Flight Delay and Cancellation Dataset (2019–2023)](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
enriched with daily weather data from the [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api).

**Contributors:** Sarah Menezes, Ved Patel, Aditya Velagapudi

---

## Project Overview

Flight delays cost the U.S. economy billions of dollars each year and affect millions of passengers. This project builds three machine learning models that predict — before a flight departs — whether that flight will be delayed by more than 15 minutes.

The key insight is that **many common causes of delays are predictable ahead of time**: bad weather at the origin airport, historically poor-performing airlines, and congested routes all leave a signal in the data before wheels-up.

---

## How to Run (Step by Step)

### 0. Install dependencies
```bash
pip install -r requirements.txt
```

### 1. Clean the raw data
```bash
python3 clean_data.py
```
Reads `flights_sample_3m.csv`, creates the target column, removes leaky columns, extracts date/time features, and saves `cleaned_flights.csv`.

### 2. Fetch weather data
```bash
python3 fetch_weather.py
```
Calls the Open-Meteo Archive API for every unique origin airport in the dataset. Results are cached in `weather_cache.csv` — **re-running is safe and fast** (already-fetched airports are skipped).

> **Note:** The dataset covers ~200+ airports across 2019–2023. The full fetch takes roughly 5–10 minutes on first run, then is instant on subsequent runs.

### 3. Merge weather into flights
```bash
python3 merge_weather.py
```
Joins `cleaned_flights.csv` with `weather_cache.csv` on `ORIGIN + FL_DATE` and saves `flights_with_weather.csv`.

### 4. Run the models
```bash
python3 model_logistic_regression.py
python3 model_random_forest.py
python3 model_gradient_boosting.py
```
Each model file is self-contained. They all read `flights_with_weather.csv` and print evaluation metrics to the terminal, plus save charts as PNG files.

---

## Dataset

| File | Description |
|---|---|
| `flights_sample_3m.csv` | Raw input — 3M individual U.S. flight records, 2019–2023 |
| `cleaned_flights.csv` | Cleaned output from `clean_data.py` |
| `weather_cache.csv` | Daily weather per airport from Open-Meteo |
| `flights_with_weather.csv` | Final enriched dataset used by all models |
| `airports.csv` | Lookup table: airport IATA code → latitude, longitude |

---

## Features Used

All features below are knowable **before** the flight departs (non-leaky).

| Feature | Description |
|---|---|
| `YEAR`, `MONTH`, `DAY` | Calendar date |
| `DAY_OF_WEEK` | 0=Monday … 6=Sunday |
| `IS_WEEKEND` | 1 if Saturday or Sunday |
| `SCHEDULED_DEP_HOUR` | Hour of scheduled departure (0–23) |
| `DEP_TIME_BLOCK` | Broad time period (overnight / morning / afternoon / evening / night) |
| `CRS_ELAPSED_TIME` | Scheduled flight duration in minutes |
| `DISTANCE` | Distance between origin and destination in miles |
| `AIRLINE_CODE` | 2-letter airline code (e.g. UA, DL, AA) |
| `ORIGIN` | Origin airport IATA code (e.g. ATL, LAX) |
| `DEST` | Destination airport IATA code |
| `temperature_2m_max` | Max temperature at origin airport that day (°C) |
| `temperature_2m_min` | Min temperature at origin airport that day (°C) |
| `precipitation_sum` | Total rainfall/snowfall at origin that day (mm) |
| `wind_speed_10m_max` | Max wind speed at origin that day (km/h) |

### Target Column
`DELAYED = 1` if departure delay > 15 minutes OR flight was cancelled, else `0`.
The 15-minute threshold is the FAA/BTS official definition of a flight delay.

---

## What Columns Were Excluded (and Why)

These columns are **leaky** — they contain information that is only available after the flight has already happened. Using them as features would let the model "cheat" and give falsely high accuracy.

| Column | Why it's leaky |
|---|---|
| `DEP_TIME` | Actual departure time — only known after departure |
| `DEP_DELAY` | The delay itself — this IS what we're predicting |
| `ARR_TIME`, `ARR_DELAY` | Arrival outcomes — only known after landing |
| `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN` | All post-departure events |
| `CANCELLED`, `CANCELLATION_CODE` | Post-flight outcomes |
| `DIVERTED` | Post-flight outcome |
| `ELAPSED_TIME`, `AIR_TIME` | Actual durations — only known after landing |
| `DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, etc. | Delay cause codes filled in after landing |

---

## Models

### 1. Logistic Regression (`model_logistic_regression.py`)
A linear classifier. Learns a weight (coefficient) for each feature and passes the weighted sum through a sigmoid function to produce a probability. Fast to train and easy to interpret — the sign and magnitude of each coefficient directly shows how a feature pushes the prediction toward or away from "delayed."

- Uses `StandardScaler` (required for logistic regression)
- Solver: `saga` (fast for large datasets)
- Default: trains on 500K row sample for speed

### 2. Random Forest (`model_random_forest.py`)
Builds 200 decision trees, each on a different random sample of the data. Predictions are decided by majority vote across all trees. Naturally handles non-linear relationships (e.g., precipitation matters much more at northern airports in winter).

- No feature scaling required
- Outputs feature importance chart (`rf_feature_importance.png`)
- Default: trains on 500K row sample for speed

### 3. Gradient Boosting (`model_gradient_boosting.py`)
Builds trees sequentially — each tree corrects the errors of all previous trees. Uses `HistGradientBoostingClassifier`, sklearn's fast histogram-based implementation designed for millions of rows.

- **Early stopping** automatically determines the optimal number of trees
- Can train on the full 3M rows in a few minutes
- Outputs feature importance chart and learning curve

---

## Evaluation Metrics

All three models report:

| Metric | What it means |
|---|---|
| **Accuracy** | % of all predictions that were correct |
| **Precision** | Of flights predicted as delayed, how many actually were |
| **Recall** | Of flights that actually were delayed, how many the model caught |
| **F1 Score** | Harmonic mean of precision and recall (good balance metric) |
| **ROC-AUC** | Area under the ROC curve; 1.0 = perfect, 0.5 = random guessing |
| **Confusion Matrix** | Breakdown of true positives, false positives, true negatives, false negatives |

> **Tip:** For flight delay prediction, **Recall** is often the most important metric — missing a real delay (false negative) is more costly to passengers than a false alarm.

---

## How Open-Meteo Weather Is Joined

1. `fetch_weather.py` reads the list of unique `ORIGIN` airports and the date range from `cleaned_flights.csv`.
2. For each airport, it looks up the latitude and longitude in `airports.csv`.
3. It calls the Open-Meteo Archive API once per airport for the full date range (e.g., 2019-01-01 to 2023-12-31), returning one row of weather per day.
4. `merge_weather.py` performs a **left join** on `(ORIGIN, FL_DATE)` — every flight gets the weather conditions at its departure airport on its departure date.
5. Flights at airports not found in `airports.csv` receive `NaN` for weather features, which are filled with the column median before modeling.

---

## Technologies Used
- Python 3.8+
- pandas, numpy
- scikit-learn
- requests (Open-Meteo API)
- matplotlib
