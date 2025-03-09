import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

# Simulated German data (REPLACE WITH ACTUAL DATA)
np.random.seed(123)  # Different seed for Germany
dates = pd.date_range(start=datetime(2000, 1, 1), end=datetime(2024, 1, 1), freq='YS') #Yearly data
gdp_change_germany = np.random.normal(0.01, 0.015, len(dates))  # Simulated GDP changes
unemployment_change_germany = -0.2 * gdp_change_germany + np.random.normal(0, 0.008, len(dates)) #Simulated Unemployment changes, notice -0.2

data_germany = pd.DataFrame({
    'GDP_Change_Germany': gdp_change_germany,
    'Unemployment_Change_Germany': unemployment_change_germany
}, index=dates)

# Regression Analysis for Germany
X_germany = data_germany['GDP_Change_Germany']
y_germany = data_germany['Unemployment_Change_Germany']
X_germany = sm.add_constant(X_germany)
model_germany = sm.OLS(y_germany, X_germany).fit()

print("Germany Regression Results:")
print(model_germany.summary())

# Plotting German data
plt.figure(figsize=(12, 6))
plt.scatter(data_germany['GDP_Change_Germany'], data_germany['Unemployment_Change_Germany'])
plt.plot(data_germany['GDP_Change_Germany'], model_germany.predict(X_germany), color='red', label='Regression Line')
plt.xlabel('GDP Change (Germany)')
plt.ylabel('Unemployment Change (Germany)')
plt.title("Germany: Unemployment Change vs. GDP Change")
plt.legend()
plt.grid(True)
plt.show()