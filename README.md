**Overview:**

Flight delays affect millions of travelers each year, causing disruptions for passengers, airlines, and airports. This project explores patterns in U.S. flight delay data and applies data science and machine learning techniques to better understand why delays occur and how to predict them. 
The goal is to analyze historical flight data, identify the most influential factors contributing to delays, and build models that estimate the likelihood of a flight being delayed. 

**Objectives:**
- Identify key factors that influence flight days
- Analyze how delays vary across airlines, airports, and time
- Explore patterns in different types of delays (weather, carrier, etc.)
- Engineer meaningful features from raw flight data
- Build and evaluate predictive models for delay likelihood

**Research Questions:**
- What factors most strongly influence flight delays?
- How do delays vary across airlines and airports?

**Dataset:**
- Source: Kaggle - _Airline Delays_ by Eugeniy Osetrov
- Size: ~3000 records, 21 variables
- Type: Aggregated flight delay statistics (U.S. airlines)

**Key Features:**

Time information
- year, month

Airline & Airport
- carrier, carrier_name
- airport, airport_name

Flight Activity
- arr_flights, arr_cancelled, arr_diverted

Delay Counts
- arr_del15, carrier_ct, weather_ct, nas_ct, late_aircraft_ct

Delay Times (minutes)
- arr_delay, carrier_delay, weather_delay, nas_delay, late_aircraft_delay

**Data Cleaning & Preprocessing:**

We performed several preprocessing steps to ensure data quality:
- Removed irrelevant columns: security_ct, security_delay
- Dropped rows with missing values
- Created new features: Delay Rate (proportion of delayed flights), Average Airline Delay Rate, Average Airport Delay Rate

**Methodology:**
1. Exploratory Data Analysis (EDA)
     - Analyze delay distributions
     - Compare delays across airlines and airports
     - Visualize trends over time
2. Feature Engineering
     - Time-based features (month, potential time categories)
     - Aggregated delay metrics
     - Delay rates and averages
3. Modeling Approaches
     - We plan to experiment with Logistic Regression (baseline model) and Random Forest Classifier (primary model)

**Evaluation Metrics:**

Models will be evaluated using:
- Accuracy: overall correctness
- Precision: correctness of delay predictions
- Recall: ability to detect actual delays
- F1 Score: balance between precision and recall
- Confusion Matrix: breakdown of prediction errors

**Technologies Used:**
- Python
- Pandas

**Contributors:**
Sarah Menezes, Ved Patel, Aditya Velagapudi
