import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

# Simulated data (replace with actual Canadian GDP and unemployment data)
np.random.seed(42)
dates = pd.date_range(start=datetime(2000, 1, 1), end=datetime(2024, 1, 1), freq='QS')
gdp_change = np.random.normal(0.01, 0.02, len(dates))
unemployment_change = -0.4 * gdp_change + np.random.normal(0, 0.01, len(dates))

data = pd.DataFrame({
    'GDP_Change': gdp_change,
    'Unemployment_Change': unemployment_change
}, index=dates)

# Manual Calculation of Beta (for demonstration)
delta_y = data['GDP_Change'].values
delta_u = data['Unemployment_Change'].values

mean_delta_y = np.mean(delta_y)
mean_delta_u = np.mean(delta_u)

numerator = np.sum((delta_y - mean_delta_y) * (delta_u - mean_delta_u))
denominator = np.sum((delta_y - mean_delta_y) ** 2)

beta_manual = numerator / denominator
alpha_manual = mean_delta_u - beta_manual * mean_delta_y

print("Manual Calculation:")
print("Alpha (Intercept):", alpha_manual)
print("Beta (Okun's Coefficient):", beta_manual)

# Using statsmodels (for comparison and statistical analysis)
X = data['GDP_Change']
y = data['Unemployment_Change']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print("\nStatsmodels Calculation:")
print(model.summary())

# Plotting
plt.figure(figsize=(12, 6))
plt.scatter(data['GDP_Change'], data['Unemployment_Change'])
plt.plot(data['GDP_Change'], model.predict(X), color='red', label='Regression Line')
plt.xlabel('GDP Change')
plt.ylabel('Unemployment Change')
plt.title("Unemployment Change vs. GDP Change")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Unemployment_Change'], label='Unemployment Change')
plt.plot(data.index, -0.4 * data['GDP_Change'], label='-0.4 * GDP Change', alpha=0.5)
plt.legend()
plt.title("Unemployment Change Over Time")
plt.grid(True)
plt.show()