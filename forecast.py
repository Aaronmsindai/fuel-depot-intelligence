import pandas as pd
from sqlalchemy import create_engine, text
import getpass
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# 1. Connect to Database
DB_USER = getpass.getuser()
engine = create_engine(f"postgresql://{DB_USER}@localhost:5432/fuel_db")

# 2. Extract Daily Sales for Diesel
query = """
SELECT 
    transaction_date, 
    SUM(volume_liters) as total_liters
FROM fact_fuel_transactions
WHERE transaction_type = 'Sale' AND fuel_type = 'Diesel'
GROUP BY transaction_date
ORDER BY transaction_date;
"""

print("Fetching data from PostgreSQL...")
df = pd.read_sql(query, engine)

# Set date as index for time-series
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df.set_index('transaction_date', inplace=True)

# Fill missing days with 0 (if no sales happened that day)
df = df.asfreq('D').fillna(0)

# 3. Train SARIMA Model
print("Training SARIMA model...")
# (p,d,q) = (1,1,1), (P,D,Q,s) = (1,1,1,7) for weekly seasonality
model = SARIMAX(df['total_liters'], order=(1,1,1), seasonal_order=(1,1,1,7))
results = model.fit(disp=False)

# 4. Forecast next 7 days
print("Forecasting next 7 days...")
forecast = results.get_forecast(steps=7)
forecast_df = forecast.summary_frame()

# 5. Plot the results
print("\n--- FORECAST FOR NEXT 7 DAYS ---")
print(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']])

plt.figure(figsize=(12, 6))
plt.plot(df.index[-60:], df['total_liters'][-60:], label='Historical Sales (Last 60 Days)')
plt.plot(forecast_df.index, forecast_df['mean'], label='Forecast', color='red')
plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
plt.title('Diesel Sales Forecast')
plt.legend()
plt.show()