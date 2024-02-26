# %%


import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

class Trend(BaseData):
    # Assuming you would define a Trend class similar to Seasonal
    ...


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
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.intercepts = []
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
        self.noise_mean = noise_mean
        self.noise_std = noise_std
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
    def __init__(self, t_0, t_max, t_delta=1):
        self.t_0 = t_0
        self.t_max = t_max
        self.t_delta = t_delta
        self.components = []

    def add_component(self, component):
        if isinstance(component, (Seasonal, Trend, ContinuousPiecewiseLinearFunction)):
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

    def plot(self):
        pd.Series(self.data).plot()
        plt.show()


# %%

_slopes = [1.2, -0.1, -0.3]
_changepoints = [75, 95, 150]
_size = 225


cplf = ContinuousPiecewiseLinearFunction(t_max=_size,
    slopes=_slopes, changepoints=_changepoints, noise=True, noise_std=1, noise_mean=10
)

# %%



t1 = Seasonal(t_max=_size, amplitude=10, period=20, noise=False, name='weather')
t2 = Seasonal(t_max=_size, amplitude=2, period=100, noise=True, noise_mean=0, noise_std=0.1, name='market')

y = pd.Series(t1.generate_data()[1])

# %%


dm = DecomposableModel(t_0=0, t_max=_size)
dm.add_component(t1)
dm.add_component(t2)
dm.add_component(cplf)
dm.list_components()

#%%

dm._update_data()
dm.plot()
# %%

sr = pd.Series(cplf.generate_sample_data_fixed_interval()[1])
df = pd.DataFrame()
df['target'] = sr.values
(df['target'] + y).plot()
plt.show()

# %%
