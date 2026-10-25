import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import getpass
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Page Config ---
st.set_page_config(page_title="Fuel Depot Dashboard", page_icon="⛽", layout="wide")
st.title("⛽ Fuel Depot Intelligence Dashboard")
st.markdown("Live inventory tracking, revenue analysis, and machine learning forecasting.")

# --- 2. Database Connection ---
@st.cache_resource
def get_engine():
    DB_USER = getpass.getuser()
    return create_engine(f"postgresql://{DB_USER}@localhost:5432/fuel_db")

engine = get_engine()

# --- 3. Fetch Data ---
@st.cache_data(ttl=60)
def load_data():
    query = "SELECT * FROM fact_fuel_transactions;"
    df = pd.read_sql(query, engine)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

df = load_data()

# --- 4. KPI Metrics (Top Row) ---
st.subheader("📊 Overall Business Metrics")
total_revenue = df[df['transaction_type'] == 'Sale']['total_revenue'].sum()
total_delivered = df[df['transaction_type'] == 'Delivery']['volume_liters'].sum()
total_sold = df[df['transaction_type'] == 'Sale']['volume_liters'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue ($)", f"${total_revenue:,.2f}")
col2.metric("Total Fuel Delivered (Liters)", f"{total_delivered:,.0f} L")
col3.metric("Total Fuel Sold (Liters)", f"{total_sold:,.0f} L")

st.divider()

# --- 5. Revenue by Fuel Type ---
st.subheader("💰 Revenue by Fuel Type")
revenue_by_fuel = df[df['transaction_type'] == 'Sale'].groupby('fuel_type')['total_revenue'].sum().reset_index()
fig_pie = px.pie(revenue_by_fuel, values='total_revenue', names='fuel_type', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# --- 6. Machine Learning Forecasting ---
st.subheader("🔮 7-Day Sales Forecast (SARIMA Model)")
st.markdown("Select a fuel type to see the Machine Learning forecast based on historical daily sales.")

fuel_selection = st.selectbox("Select Fuel Type", ['Diesel', 'Petrol', 'Kerosene'])

with st.spinner("Training SARIMA model and generating forecast..."):
    # Prepare data for selected fuel
    query = f"SELECT transaction_date, SUM(volume_liters) as total_liters FROM fact_fuel_transactions WHERE transaction_type = 'Sale' AND fuel_type = '{fuel_selection}' GROUP BY transaction_date ORDER BY transaction_date;"
    ts_df = pd.read_sql(query, engine)
    ts_df['transaction_date'] = pd.to_datetime(ts_df['transaction_date'])
    ts_df.set_index('transaction_date', inplace=True)
    ts_df = ts_df.asfreq('D').fillna(0)
    
    # Train Model
    model = SARIMAX(ts_df['total_liters'], order=(1,1,1), seasonal_order=(1,1,1,7))
    results = model.fit(disp=False)
    
    # Forecast
    forecast = results.get_forecast(steps=7)
    forecast_df = forecast.summary_frame()

# Plot Forecast
fig = go.Figure()

# Historical Data (Last 30 days for clean view)
fig.add_trace(go.Scatter(x=ts_df.index[-30:], y=ts_df['total_liters'][-30:], mode='lines', name='Historical Sales', line=dict(color='blue')))

# Forecast Line
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast', line=dict(color='red')))

# Confidence Interval (Shaded Area)
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

st.caption("Dashboard created by Aaron Msindai | Powered by Streamlit, PostgreSQL, and Statsmodels")