import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Data from the table
data_india = pd.DataFrame({
    'Year': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Real_GDP_Growth': [3.8, 4.8, 3.8, 7.9, 7.9, 9.3, 9.3, 7.7, 3.1, 7.9, 8.5, 5.2, 5.5, 6.4, 7.4, 8.0, 8.2, 6.8, 6.5, 3.7, -6.6, 8.7, 6.8, 7.2],
    'Unemployment_Rate': [8.3, 8.8, 8.3, 8.1, 8.0, 7.9, 7.8, 7.7, 8.3, 8.8, 8.5, 8.2, 7.9, 7.8, 7.7, 7.7, 7.8, 8.0, 8.3, 8.8, 10.0, 9.0, 8.5, 8.2]
})

# Calculate annual changes
data_india['GDP_Change'] = data_india['Real_GDP_Growth']
data_india['Unemployment_Change'] = data_india['Unemployment_Rate'].diff()

# Perform regression for the entire period
subset = data_india.dropna()  # Remove rows with NaN (from the first diff)

X = subset['GDP_Change']
y = subset['Unemployment_Change']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print("India Regression Results (2000-2023):")
print(model.summary())

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(subset['GDP_Change'], subset['Unemployment_Change'])
plt.plot(subset['GDP_Change'], model.predict(X), color='red', label='Regression Line')
plt.xlabel('GDP Growth (%)')
plt.ylabel('Unemployment Change (%)')
plt.title("India: Unemployment vs. GDP (2000-2023)")
plt.legend()
plt.grid(True)
plt.show()