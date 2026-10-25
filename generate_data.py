import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)

# 1. Setup parameters
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
days = (end_date - start_date).days

fuel_types = ['Petrol', 'Diesel', 'Kerosene']
trucks = [f"TRK-{i}" for i in range(100, 115)]
data = []

# 2. Generate daily transactions (Deliveries and Sales)
print("Generating data...")
for day in range(days):
    current_date = start_date + timedelta(days=day)
    
    for fuel in fuel_types:
        # Simulate 1 to 3 deliveries per day per fuel type
        for _ in range(random.randint(1, 3)):
            data.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Transaction_Type': 'Delivery',
                'Fuel_Type': fuel,
                'Volume_Liters': random.randint(5000, 15000),
                'Price_Per_Liter': round(random.uniform(1.5, 2.5), 2),
                'Truck_ID': random.choice(trucks)
            })
            
        # Simulate 5 to 10 customer sales per day per fuel type
        for _ in range(random.randint(5, 10)):
            data.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Transaction_Type': 'Sale',
                'Fuel_Type': fuel,
                'Volume_Liters': random.randint(500, 3000),
                'Price_Per_Liter': round(random.uniform(2.5, 3.5), 2),
                'Truck_ID': None
            })

# 3. Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv('fuel_transactions.csv', index=False)
print(f"Data generated successfully! {len(df)} rows saved to fuel_transactions.csv")