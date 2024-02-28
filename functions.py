# %%


import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

from prophet import Prophet


class BaseData:
    def __init__(self, noise=False, noise_mean=0, noise_std=1):
        self.noise = noise
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.data = None

    def add_noise(self, data):
        noise = np.random.normal(loc=self.noise_mean, scale=self.noise_std, size=len(data))
        data += noise
        return data

    def plot(self):
        pd.Series(self.data[1]).plot()
        plt.show()


class ContinuousPiecewiseLinearFunction(BaseData):

    def __init__(self, t_max, slopes: list, changepoints: list, t_0=0, t_delta=1, noise=False, noise_mean=0,
                 noise_std=1, name=None):
        super().__init__(noise, noise_mean, noise_std)
        self.slopes = slopes
        self.changepoints = changepoints
        self.end_time = t_max
        self.start_time = t_0
        self.time_interval = t_delta
        self.noise = noise
        self.intercepts = []
        self.name = name
        self.data = self.generate_data()

    def calculate_intercept(self):
        self.intercepts = [0]  # intercept is 0
        for i in range(1, len(self.slopes)):
            # calculate intercept such that the function is continuous at changepoints
            intercept = self.intercepts[-1] + self.slopes[i - 1] * self.changepoints[i - 1] - self.slopes[i] * \
                        self.changepoints[i - 1]
            self.intercepts.append(intercept)

    def piecewise_linear(self, x):
        for i, cp in enumerate(self.changepoints):
            if x <= cp:
                return self.slopes[i] * x + self.intercepts[i]
        return self.slopes[-1] * x + self.intercepts[-1]

    def generate_data(self):
        self.calculate_intercept()
        x_values = [x for x in range(self.start_time, self.end_time + 1, self.time_interval)]
        y_values = [self.piecewise_linear(x) for x in x_values]
        if self.noise:
            y_values = self.add_noise(y_values)
        return x_values, y_values


class Seasonal(BaseData):

    def __init__(
            self, t_max, amplitude, period, t_0=0, t_delta=1, noise=False, noise_mean=0, noise_std=1, name=None):
        super().__init__(noise, noise_mean, noise_std)
        self.name = name
        self.t_0 = t_0
        self.t_max = t_max
        self.t_delta = t_delta
        self.amplitude = amplitude
        self.period = period
        self.data = self.generate_data()

    def __repr__(self):
        return f"SeasonalFunction(name={self.name}, pattern={None})"

    def generate_data(self):
        x = np.arange(self.t_0, self.t_max + 1, self.t_delta)
        y = self.amplitude * np.sin(2 * np.pi * x / self.period)
        if self.noise:
            y = self.add_noise(y)
        return x, y


class DecomposableModel:
    def __init__(self, t_0, t_max, index='index', t_delta=1):
        self.index = index
        self.t_0 = t_0
        self.t_max = t_max
        self.t_delta = t_delta
        self.components = []

    def add_component(self, component):
        if isinstance(component, (Seasonal, ContinuousPiecewiseLinearFunction)):
            self.components.append(component)
        else:
            raise ValueError("Component must be an instance of Seasonal or Trend.")

    def remove_component(self, component):
        self.components.remove(component)

    def list_components(self):
        for component in self.components:
            print(component)

    def _update_data(self):
        data = pd.Series(np.repeat(0, self.t_max + 1))
        for component in self.components:
            data += component.data[1]
        self.data = data.copy(deep=True)

    def export(self, index: str = None, index_map: dict = None):
        if index and index_map:
            data = self.data.copy(deep=True)
            data.index = data.index.map(index_map[(self.index, index)])
        return data

    def plot(self):
        pd.Series(self.data).plot()
        plt.show()


class TimeSeries:
    # TODO: make it so that on TimeSeries we can hang multiple predictive models and models datasets
    # E.g., multiple data preprocessing methods can be compared

    def __init__(self, index, index_map, data_future):
        self.index = index
        self.index_map = index_map
        self.data_historic = dict()
        self.data_exogenous = dict()
        self.data_future = data_future
        self.models = dict()
        self.data_forecast = dict()

    def add_data_historic(self, name, data_historic):
        """Adds a new dataset to the TimeSeries object.t
        """
        self.data_historic[name] = data_historic

    def add_data_exogenous(self, name, data_exogenous):
        """Adds a new dataset to the TimeSeries object.t
        """
        self.data_exogenous[name] = data_exogenous

    def add_model(self, name, model):
        """Adds a new predictive model to the TimeSeries object.
        """
        if not issubclass(model.__class__, Model):
            raise TypeError(f"The model {name} must be a subclass of Model")
        self.models[name] = model
        self.models[name] = model

    def fit(self, model_name, data_name, **fit_params):
        """Fit a model to a specified variation of the time series data."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        if data_name not in self.data_historic:
            raise ValueError(f"Data {data_name} not found.")

        model = self.models[model_name]
        data = self.data_historic[data_name]
        model.fit(data, **fit_params)  # Assuming the model has a fit method

    def forecast(self, model_name, periods, **forecast_params):
        """Generate forecasts using a specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")

        model = self.models[model_name]
        forecast = model.forecast(periods, **forecast_params)
        self.data_forecast[model_name] = forecast
        return self.data_forecast


    def plot_all(self):
        fig = go.Figure()
        future = self.data_future.copy(deep=True)
        future = future.reset_index()
        future.columns = ['ds', 'y']
        for name, data in self.data_historic.items():
            if isinstance(data, pd.DataFrame) and 'ds' in data and 'y' in data:
                fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name=f"Data: {name}"))
            else:
                fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name=f"Data: {name}"))

        for model_name, forecast in self.data_forecast.items():
            if isinstance(forecast, pd.DataFrame) and 'ds' in forecast and 'yhat' in forecast:
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name=f"Forecast: {model_name}", line=dict(dash='dash')))
        if isinstance(future, pd.DataFrame) and 'ds' in future and 'y' in future:
            fig.add_trace(go.Scatter(x=future['ds'], y=future['y'], mode='lines+markers', name=f"Future", line=dict(dash='dash')))

        fig.update_layout(title='Time Series Data Variations and Forecasts',
                          xaxis_title='Time',
                          yaxis_title='Value',
                          legend_title='Legend')
        fig.show()

class Model:
    # Base model
    def __init__(self):
        pass

    def fit(self, data, **fit_params):
        raise NotImplementedError("Subclasses must implement the fit method.")

    def forecast(self, periods, **forecast_params):
        raise NotImplementedError("Subclasses must implement the forecast method.")




class ProphetModel(Model):
    def __init__(self, **prophet_params):
        super().__init__()
        self.model: Prophet = Prophet(**prophet_params)
        self.fitted = False

    def fit(self, data, **fit_params):
        """Fit the Prophet model to the time series data.

        Parameters:
        - data: A pandas DataFrame with columns 'ds' and 'y'.
        - fit_params: Additional parameters to pass to the Prophet fit method.
        """
        data = data.reset_index()
        data.columns = ['ds', 'y']
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame with 'ds' and 'y' columns.")

        # Ensure 'ds' is datetime type and 'y' is float
        if 'ds' not in data.columns or 'y' not in data.columns:
            raise ValueError("DataFrame must contain 'ds' and 'y' columns.")

        self.model.fit(data, **fit_params)
        self.fitted = True

    def forecast(self, periods, frequency='W', **forecast_params):
        """Generate future forecasts using the fitted Prophet model.

        Parameters:
        - periods: An integer specifying the number of periods to forecast.
        - frequency: A string specifying the frequency of the forecast ('D' for daily, 'W' for weekly).
        - forecast_params: Additional parameters to pass to the Prophet predict method.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting.")

        future = self.model.make_future_dataframe(periods=periods, freq=frequency, **forecast_params)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat']]  # Return a DataFrame with the forecast


