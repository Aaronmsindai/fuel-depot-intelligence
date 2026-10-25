import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Page Config ---
st.set_page_config(page_title="Fuel Depot Dashboard", page_icon="⛽", layout="wide")
st.title("⛽ Fuel Depot Intelligence Dashboard")
st.markdown("Live inventory tracking, revenue analysis, and machine learning forecasting.")

# --- 2. Load Data from CSV ---
@st.cache_data
def load_data():
    # Read the CSV file directly from the repo
    df = pd.read_csv("fuel_transactions.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# --- 3. KPI Metrics (Top Row) ---
st.subheader("📊 Overall Business Metrics")
total_revenue = df[df['Transaction_Type'] == 'Sale'].apply(lambda x: x['Volume_Liters'] * x['Price_Per_Liter'], axis=1).sum()
total_delivered = df[df['Transaction_Type'] == 'Delivery']['Volume_Liters'].sum()
total_sold = df[df['Transaction_Type'] == 'Sale']['Volume_Liters'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue ($)", f"${total_revenue:,.2f}")
col2.metric("Total Fuel Delivered (Liters)", f"{total_delivered:,.0f} L")
col3.metric("Total Fuel Sold (Liters)", f"{total_sold:,.0f} L")

st.divider()

# --- 4. Revenue by Fuel Type ---
st.subheader("💰 Revenue by Fuel Type")
sales_df = df[df['Transaction_Type'] == 'Sale'].copy()
sales_df['Revenue'] = sales_df['Volume_Liters'] * sales_df['Price_Per_Liter']
revenue_by_fuel = sales_df.groupby('Fuel_Type')['Revenue'].sum().reset_index()

fig_pie = px.pie(revenue_by_fuel, values='Revenue', names='Fuel_Type', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# --- 5. Machine Learning Forecasting ---
st.subheader("🔮 7-Day Sales Forecast (SARIMA Model)")
st.markdown("Select a fuel type to see the Machine Learning forecast based on historical daily sales.")

fuel_selection = st.selectbox("Select Fuel Type", ['Diesel', 'Petrol', 'Kerosene'])

with st.spinner("Training SARIMA model and generating forecast..."):
    # Prepare data for selected fuel from CSV
    ts_df = sales_df[sales_df['Fuel_Type'] == fuel_selection]
    ts_df = ts_df.groupby('Date')['Volume_Liters'].sum().reset_index()
    ts_df.set_index('Date', inplace=True)
    ts_df = ts_df.asfreq('D').fillna(0)
    
    # Train Model
    model = SARIMAX(ts_df['Volume_Liters'], order=(1,1,1), seasonal_order=(1,1,1,7))
    results = model.fit(disp=False)
    
    # Forecast
    forecast = results.get_forecast(steps=7)
    forecast_df = forecast.summary_frame()

# Plot Forecast
fig = go.Figure()

# Historical Data (Last 30 days)
fig.add_trace(go.Scatter(x=ts_df.index[-30:], y=ts_df['Volume_Liters'][-30:], mode='lines', name='Historical Sales', line=dict(color='blue')))

# Forecast Line
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast', line=dict(color='red')))

# Confidence Interval
fig.add_trace(go.Scatter(
    x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
    y=forecast_df['mean_ci_upper'].tolist() + forecast_df['mean_ci_lower'].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.update_layout(title=f"{fuel_selection} Sales Forecast", xaxis_title="Date", yaxis_title="Volume (Liters)")
st.plotly_chart(fig, use_container_width=True)

st.caption("Dashboard created by Aaron Msindai | Powered by Streamlit, Pandas, and Statsmodels")
st.divider()
st.subheader("🤖 Explainable AI: Why did the model predict this?")
st.markdown("Using SHAP (SHapley Additive exPlanations) to understand which factors drive Diesel sales the most.")

import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Prepare data for Gradient Boosting
# We will specifically look at Diesel for this explanation
diesel_sales = sales_df[sales_df['Fuel_Type'] == 'Diesel'].copy()
daily_sales = diesel_sales.groupby('Date')['Volume_Liters'].sum().reset_index()
daily_sales = daily_sales.sort_values('Date')

# Feature Engineering
daily_sales['Day_of_Week'] = daily_sales['Date'].dt.day_of_week
daily_sales['Month'] = daily_sales['Date'].dt.month
daily_sales['Is_Weekend'] = daily_sales['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
daily_sales['Yesterday_Sales'] = daily_sales['Volume_Liters'].shift(1)
daily_sales['Last_Week_Sales'] = daily_sales['Volume_Liters'].shift(7)
daily_sales = daily_sales.dropna()

features = ['Day_of_Week', 'Month', 'Is_Weekend', 'Yesterday_Sales', 'Last_Week_Sales']
X = daily_sales[features]
y = daily_sales['Volume_Liters']

# 2. Train the model
model_gb = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42)
model_gb.fit(X, y)

# 3. Generate SHAP plot
with st.spinner("Calculating SHAP values (this takes a few seconds)..."):
    explainer = shap.Explainer(model_gb, X)
    shap_values = explainer(X)
    
    fig_shap, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig_shap)