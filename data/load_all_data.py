"""
Load all data files at once
Run this to populate all data folders
"""

import os
import pandas as pd

# Create data directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Load retail sales data
retail_sales = pd.read_csv('data/raw/retail_sales_data.csv')
print(f"✅ Loaded retail sales: {len(retail_sales)} rows")

# Load product master
product_master = pd.read_csv('data/raw/product_master.csv')
print(f"✅ Loaded products: {len(product_master)} rows")

# Load store master
store_master = pd.read_csv('data/raw/store_master.csv')
print(f"✅ Loaded stores: {len(store_master)} rows")

# Load calendar
calendar = pd.read_csv('data/raw/calendar_data.csv')
print(f"✅ Loaded calendar: {len(calendar)} rows")

print("\n✅ All data files loaded successfully!")