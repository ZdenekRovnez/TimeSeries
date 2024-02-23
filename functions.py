# %%


import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_noise(data, noise_mean, noise_std):
    noise = np.random.normal(loc=noise_mean, scale=noise_std, size=len(data))
    data += noise
    return data


class BaseData:
    def __init__(self, add_noise_param=False):
        self.add_noise_param = add_noise_param

    def add_normal_noise_to_data(self, data, noise_mean, noise_std):
        data = add_noise(data, noise_mean, noise_std)
        return data


class WaveFunction(BaseData):

    def __init__(self, size, _add_noise):
        super().__init__(_add_noise)
        self.test = 0
        self.end_time = size
        self.start_time = 0  # Start at zero
        self.time_interval = 10  # Fixed time interval of 1
        self.noise_level = 0

    def generate_data(self):
        x_values = [x for x in range(self.start_time, self.end_time + 1, self.time_interval)]
        y_values = [50 * np.sin(x / 20) for x in x_values]
        if self.add_noise_param:
            y_values = self.add_normal_noise_to_data(y_values, 5, 10)
        return x_values, y_values

class ContinuousPiecewiseLinearFunction(BaseData):

    def __init__(self, slopes: list, changepoints: list, size, _add_noise):
        super().__init__(_add_noise)
        self.slopes = slopes
        self.changepoints = changepoints
        self.end_time = size
        self.start_time = 0  # Start at zero
        self.time_interval = 10  # Fixed time interval of 1
        self.noise_level = 0
        self.intercepts = []

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

    def generate_sample_data_fixed_interval(self):
        self.calculate_intercept()
        x_values = [x for x in range(self.start_time, self.end_time + 1, self.time_interval)]
        y_values = [self.piecewise_linear(x) + random.uniform(-self.noise_level, self.noise_level) for x in
                    x_values]
        if self.add_noise_param:
            y_values = self.add_normal_noise_to_data(y_values, 5, 10)
        return x_values, y_values


# %%

_slopes = [1.2, -0.1, -0.3]
_changepoints = [150, 300, 450]
_size = 600

cplf = ContinuousPiecewiseLinearFunction(
    slopes=_slopes, changepoints=_changepoints, size=_size, _add_noise=True
)

# %%


_slopes = [1.2, -0.1, -0.3]
_changepoints = [150, 300, 450]
_size = 600

t = WaveFunction(size=_size, _add_noise=True)

y = pd.Series(t.generate_data()[1])

#%%

sr = pd.Series(cplf.generate_sample_data_fixed_interval()[1])
df = pd.DataFrame()
df['target'] = sr.values
(df['target']+y).plot()
plt.show()

# %%
