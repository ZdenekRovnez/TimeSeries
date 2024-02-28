import numpy as np
import pandas as pd
# %% Get the data

from functions import Seasonal, ContinuousPiecewiseLinearFunction, DecomposableModel

# Create the data for the time series object
slopes = [1.2, -0.1, -0.3]
changepoints = [75, 95, 150]
size = 300
from utils_time_series import create_index_map

(index_map, indices) = create_index_map(t_max=size, start_date='2018-01-01')

t1 = Seasonal(t_max=size, amplitude=10, period=20, noise=False, name='seasonal1')
t2 = Seasonal(t_max=size, amplitude=2, period=100, noise=True, noise_mean=0, noise_std=0.1, name='seasonal2')
cplf = ContinuousPiecewiseLinearFunction(
    t_max=size, slopes=slopes, changepoints=changepoints, noise=True, noise_std=1, noise_mean=10
)

dm = DecomposableModel(t_0=0, t_max=size)
dm.add_component(t1)
dm.add_component(t2)
dm.add_component(cplf)
dm.list_components()
dm._update_data()

fc_horizon = 12

data = dm.export(index='timestamp', index_map=index_map)

future = data[(len(data) - fc_horizon):]
data = data[:(len(data) - fc_horizon)]

data2 = data.rolling(2).agg('mean')
data2 = data2.fillna(method='bfill')

# %% Initialize the TimeSeries
from functions import TimeSeries

from utils_time_series import create_index_map

# Create an instance of the TimeSeries class
ts = TimeSeries(index=None, index_map=index_map, data_future=future)

# Add data variations
ts.add_data_historic("original", data)
ts.add_data_historic("variation", data2)

# %% Add the models

from functions import ProphetModel

ts.add_model("prophet", ProphetModel())
ts.add_model("prophet2", ProphetModel())

# %%

# Fit models to specific data variations
ts.fit("prophet", "original")
ts.fit("prophet2", "variation")

# %%

# Generate forecasts
forecast_prophet = ts.forecast("prophet", periods=12)
forecast_prophet2 = ts.forecast("prophet2", periods=12)

# %%

ts.plot_all()

