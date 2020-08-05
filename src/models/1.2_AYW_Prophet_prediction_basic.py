# This program predicts a constant yearly power degradation rate
# using the Facebook open-source tool Prophet.
# Program is tailored to take as input a dataset generate using
# RDtool with time labeled as “Unnamed: 0” and power labeled
# as “Power.”

import pandas as pd
from fbprophet import Prophet
import sklearn.metrics
import math
import numpy as np

# Data file path to be changed accordingly
df = pd.read_csv("./data/raw/synthetic_soil/Synthetic_Soil_1.csv")

# Reshape the timestamp to a format Prophet accepts
df = df.rename(columns={'Unnamed: 0': 'time'})
df['time'] = pd.to_datetime(df['time'])
df['time'] = df['time'].dt.tz_localize(None)
time_df = df.set_index('time')

# Reduce the dataset from minute-by-minute resolution to hour-by-hour
# to alleviate computing burden
daily_df = time_df.resample('H').mean()
daily_df = daily_df.reset_index()

# Create a dataframe with columns ds and y for time and signal
# which is the input Prophet needs
input_df = daily_df[["time", "Power"]]
input_df = input_df.rename(columns={'time': 'ds', 'Power': 'y'})
input_df["ds"] = input_df["ds"].dt.tz_localize(None)

# Create a model, turning automatic seasonalities off
# and adding them separately to be able to fine-tune
# fourier_order
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    growth='linear',  # degradation is expected to continue indefinitely
    n_changepoints=0,  # constant degradation, no trend turning points
    seasonality_mode='multiplicative')  # degradation is a factor of the signal
model.add_seasonality(name="daily", period=1, fourier_order=15)
model.add_seasonality(name="yearly", period=365.25, fourier_order=1)
model.fit(input_df)

# Make a prediction, needed to output the trend
future = pd.date_range('2015-01-01', periods=daily_df.shape[0], freq='H')
future = pd.DataFrame(future)
future = future.rename(columns={0: 'ds'})
forecast = m.predict(future)

# If you want to have a look at the trend as a time series
# fig2 = m.plot_components(forecast)

# Extract yearly degradation rate from trend and compares to the label
model = np.polyfit(
    [x/24/365 for x in range(0, len(forecast))],
    forecast.trend,
    1)
print("Actual degradation: "+str(df.iloc[0].Degradation_rate_per_yr))
print("Predicted degradation: "+str(model[0]/model[1]))
