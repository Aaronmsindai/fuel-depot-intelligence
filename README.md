# ⛽ Fuel Depot Intelligence Dashboard

An end-to-end Data Science and Data Engineering pipeline designed to track fuel depot operations, analyze revenue, and forecast future fuel demand using Machine Learning.

## 🚀 The Business Problem
Fuel depots generate thousands of transactions daily (deliveries from trucks, sales to customers). Without a centralized system, it is impossible for owners to track profit margins, prevent fuel shortages, or predict how much fuel they need to order next week. This project solves that problem by providing a live analytical dashboard and a machine learning forecasting engine.

## 🏗️ Architecture
1. **Data Generation:** A Python script (`generate_data.py`) simulates 1 year of realistic depot transactions (10,520 rows) including deliveries, sales, fuel types, and pricing.
2. **Database (PostgreSQL):** The data is loaded via an ETL script (`load_to_postgres.py`) into a PostgreSQL relational database using SQLAlchemy, utilizing a star-schema approach.
3. **Machine Learning (`forecast.py`):** Extracts daily sales from the database and trains a SARIMA time-series model to forecast fuel volume for the next 7 days.
4. **Dashboard (`dashboard.py`):** An interactive Streamlit dashboard that displays live KPIs, revenue breakdowns, and the ML forecast with 95% confidence intervals.

## 🛠️ Tech Stack
- **Data Engineering:** Python, Pandas, PostgreSQL, SQLAlchemy
- **Machine Learning:** Statsmodels (SARIMA Time-Series)
- **Frontend:** Streamlit, Plotly

## 💻 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Aaronmsindai/fuel-depot-intelligence.git
cd fuel-depot-intelligence
2. Create a virtual environment and install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install pandas sqlalchemy psycopg2-binary faker statsmodels streamlit plotly
```

3. Setup the Database
Ensure PostgreSQL is running locally, then create the database:

```bash
createdb fuel_db
```

4. Generate Data and Load to Database

```bash
python generate_data.py
python load_to_postgres.py
```

5. Launch the Dashboard

```bash
python -m streamlit run dashboard.py
```

📊 Dashboard Features

· Overall Business Metrics: Total revenue, total fuel delivered, and total fuel sold.
· Revenue by Fuel Type: Interactive donut chart showing the revenue split between Petrol, Diesel, and Kerosene.
· 7-Day Sales Forecast: Select a fuel type from the dropdown to dynamically train a SARIMA model and view the forecast chart with confidence intervals.

---

Created by Aaron Msindai
