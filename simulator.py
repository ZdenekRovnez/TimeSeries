# %%

# import sys
# sys.path.append(r"C:\020_Projects\_library")

#from pythonista.utils import explore

import numpy as np
import pandas as pd

import datetime
import random

from prophet import Prophet


class ContinuousPiecewiseLinearFunction:

    def __init__(self, slopes, changepoints, size):
        self.slopes = slopes
        self.changepoints = changepoints
        self.end_time = size
        self.start_time = 0  # Start at zero
        self.time_interval = 1  # Fixed time interval of 1
        self.noise_level = 0
        self.intercepts = []

    def calculate_intercept(self):
        self.intercepts = [0]  # Starting intercept is 0
        for i in range(1, len(self.slopes)):
            # Calculate intercept such that the function is continuous at changepoints
            intercept = self.intercepts[-1] + self.slopes[i - 1] * self.changepoints[i - 1] - self.slopes[i] * self.changepoints[i - 1]
            self.intercepts.append(intercept)

    def piecewise_linear(self, x):
        for i, cp in enumerate(self.changepoints):
            if x <= cp:
                return self.slopes[i] * x + self.intercepts[i]
        return self.slopes[-1] * x + self.intercepts[-1]

    def generate_sample_data_fixed_interval(self):
        self.calculate_intercept()
        x_values = [x for x in range(self.start_time, self.end_time + 1, self.time_interval)]
        y_values = [self.piecewise_linear(x) + random.uniform(-self.noise_level, self.noise_level) for x in x_values]

        return x_values, y_values


class TimeSeriesSimulator:

    def __init__(self, freq, slopes, changepoints):
        self.df_TS = None
        self.freq = freq
        self._freq_st = {'D', 'W-Mon'}
        self.start_date = '2014-01-01'
        self.end_date = '2022-12-31'
        self.time_axis = pd.date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        self.time_axis.name = 'date'
        self.len = len(self.time_axis)
        self.global_seasonal_scalar_horizontal = 1
        self.seasonality_params = {
            'season1': [2, 4]  # resp. amplitude, nr of cycles
            , 'season2': [25, 2]
            , 'season3': [12, 8]
            , 'season4': [4, 26]

        }
        self.seasonality = dict()
        self.trend = {
            'slopes':slopes,'changepoints':changepoints

        }

    def generate_continuous_piecewise_linear_data(self):

        self.CPLF = ContinuousPiecewiseLinearFunction(
            slopes=self.trend['slopes'], changepoints=self.trend['changepoints'], size=self.len - 1
        )
        sr = pd.Series(self.CPLF.generate_sample_data_fixed_interval()[1])
        df = pd.DataFrame(index = self.time_axis)
        df['target'] = sr.values
        return df


class TimeSeriesModels:

    def __init__(self):
        self.models = None

DEBUG = True

if DEBUG:
    tss = TimeSeriesSimulator(freq='W-Mon', slopes=[1, -1, 2], changepoints=[150,300])
    tss.generate_continuous_piecewise_linear_data()
    df = tss.generate_continuous_piecewise_linear_data().reset_index()
    df = df.rename(columns={'target': 'y', 'date': 'ds'})
    changepoints = ['2017-01-01','2019-09-01']
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False, changepoints=changepoints)

    m.fit(df)

    # %%
    import matplotlib.pyplot as plt

    future = m.make_future_dataframe(periods=365, freq='W-Mon')
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    plt.show()

