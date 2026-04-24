import pandas as pd

df = pd.read_csv('airline_delay.csv', na_values = 'NA')

# Drop security delay columns (not relevant to our analysis)
df = df.drop(columns=['security_ct', 'security_delay'])

# Remove rows where all values are NaN
df = df.dropna()

# 1. First Feature: Calculate the delay rate
# Represents the proportion of flights that were delayed
df['delay_rate'] = df['arr_del15'] / df['arr_flights']

# This fills any potential NaN values (caused by division by zero if arr_flights is 0)
df['delay_rate'] = df['delay_rate'].fillna(0)

# 2. Second Feature: Average delays for airlines
airline_avg_delay = df.groupby('carrier')['delay_rate'].mean().rename('avg_airline_delay_rate')
df = df.merge(airline_avg_delay, on='carrier', how='left')

# 3. Third Feature: Average delays for airports
# We calculate the mean delay rate for each airport
airport_avg_delay = df.groupby('airport')['delay_rate'].mean().rename('avg_airport_delay_rate')
df = df.merge(airport_avg_delay, on='airport', how='left')

# Save Dataset
df.to_csv('airline_delay_cleaned.csv', index=False)
