import pandas as pd
from sqlalchemy import create_engine, text
import getpass

# 1. Connect to your local PostgreSQL database
# We use getpass to automatically grab your Mac's username (aaron)
DB_USER = getpass.getuser()
DB_NAME = "fuel_db"
DB_HOST = "localhost"
DB_PORT = "5432"

# Create the connection string (no password needed for local Mac Postgres usually)
engine = create_engine(f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 2. Read the CSV
print("Reading CSV...")
df = pd.read_csv("fuel_transactions.csv")
df['Date'] = pd.to_datetime(df['Date'])

# 3. Create the table schema (Star Schema style)
create_table_query = """
DROP TABLE IF EXISTS fact_fuel_transactions;
CREATE TABLE fact_fuel_transactions (
    transaction_id SERIAL PRIMARY KEY,
    transaction_date DATE,
    transaction_type VARCHAR(50),
    fuel_type VARCHAR(50),
    volume_liters INT,
    price_per_liter FLOAT,
    total_revenue FLOAT,
    truck_id VARCHAR(50)
);
"""

print("Creating table...")
with engine.connect() as conn:
    conn.execute(text(create_table_query))
    conn.commit()

# 4. Calculate Revenue and Load to Database
print("Loading data to PostgreSQL...")
df['Total_Revenue'] = df['Volume_Liters'] * df['Price_Per_Liter']

# Rename columns to match our SQL table
df = df.rename(columns={
    'Date': 'transaction_date',
    'Transaction_Type': 'transaction_type',
    'Fuel_Type': 'fuel_type',
    'Volume_Liters': 'volume_liters',
    'Price_Per_Liter': 'price_per_liter',
    'Total_Revenue': 'total_revenue',
    'Truck_ID': 'truck_id'
})

# Load to database (append to the table we just created)
df.to_sql('fact_fuel_transactions', engine, if_exists='append', index=False)
print("Data loaded successfully to PostgreSQL!")