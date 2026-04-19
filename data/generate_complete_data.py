"""
Generate complete synthetic retail dataset
Run this instead of copying CSV files
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_all_data():
    """Generate all necessary data files"""
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("Generating synthetic retail data...\n")
    
    # 1. Generate Product Master
    print("1️⃣  Generating product master...")
    products = {
        'item_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'product_name': ['Premium Tea', 'Coffee Powder', 'Dark Chocolate', 'Rice (Basmati)', 
                        'Almond Oil', 'Pasta', 'Olive Oil', 'Flour (Wheat)', 'Sugar', 'Salt'],
        'category': ['Beverages', 'Beverages', 'Confectionery', 'Dry Goods', 
                    'Oils & Condiments', 'Dry Goods', 'Oils & Condiments', 'Dry Goods', 'Dry Goods', 'Dry Goods'],
        'brand': ['BrewMaster', 'BrewMaster', 'ChocoPro', 'GrainMax', 'NutroLife', 
                 'ItalyBest', 'PureGold', 'BreadMaker', 'SweetLife', 'PureMineral'],
        'pack_size': ['500g', '250g', '100g', '1kg', '500ml', '500g', '500ml', '1kg', '1kg', '1kg'],
        'shelf_life_days': [365, 365, 180, 730, 365, 365, 365, 365, 730, 730],
        'supplier_id': ['S001', 'S001', 'S002', 'S003', 'S004', 'S001', 'S004', 'S003', 'S002', 'S003'],
        'unit_cost': [120, 80, 200, 60, 160, 50, 180, 45, 55, 25]
    }
    df_products = pd.DataFrame(products)
    df_products.to_csv('data/raw/product_master.csv', index=False)
    print(f"   ✅ Saved: product_master.csv ({len(df_products)} products)")
    
    # 2. Generate Store Master
    print("2️⃣  Generating store master...")
    stores = {
        'store_id': [1, 2, 3, 4, 5],
        'store_name': ['Downtown Express', 'Mall Central', 'Highway Mega', 'City Center', 'Local Hub'],
        'city': ['Mumbai', 'Bangalore', 'Pune', 'Delhi', 'Hyderabad'],
        'region': ['West', 'South', 'West', 'North', 'South'],
        'store_type': ['Express', 'Premium', 'Supermarket', 'Premium', 'Express'],
        'area_sqft': [5000, 12000, 8500, 10000, 4500],
        'footfall_index': [8.5, 9.2, 7.8, 8.8, 7.2]
    }
    df_stores = pd.DataFrame(stores)
    df_stores.to_csv('data/raw/store_master.csv', index=False)
    print(f"   ✅ Saved: store_master.csv ({len(df_stores)} stores)")
    
    # 3. Generate Calendar
    print("3️⃣  Generating calendar data...")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    dates = []
    current = start_date
    
    holidays = {
        datetime(2023, 1, 26): 'Republic Day',
        datetime(2023, 3, 8): 'Maha Shivaratri',
        datetime(2023, 3, 25): 'Holi',
        datetime(2023, 4, 4): 'Ram Navami',
        datetime(2023, 4, 14): 'Ambedkar Jayanti',
        datetime(2023, 5, 23): 'Buddha Purnima',
        datetime(2023, 8, 15): 'Independence Day',
        datetime(2023, 9, 19): 'Ganesh Chaturthi',
        datetime(2023, 9, 29): 'Navratri Start',
        datetime(2023, 10, 24): 'Diwali',
        datetime(2023, 11, 13): 'Guru Nanak Jayanti',
        datetime(2023, 12, 25): 'Christmas',
    }
    
    calendar_data = []
    while current <= end_date:
        day_of_week = current.weekday()
        week_of_year = current.isocalendar()[1]
        month = current.month
        quarter = (month - 1) // 3 + 1
        day_of_year = current.timetuple().tm_yday
        is_weekend = 1 if day_of_week >= 5 else 0
        
        season = 'Winter' if month in [12, 1, 2] else ('Spring' if month in [3, 4, 5] else ('Summer' if month in [6, 7, 8] else 'Autumn'))
        
        is_holiday = 1 if current in holidays else 0
        holiday_name = holidays.get(current, '')
        
        calendar_data.append({
            'date': current.strftime('%Y-%m-%d'),
            'day_of_week': day_of_week,
            'week_of_year': week_of_year,
            'month': month,
            'quarter': quarter,
            'day_of_year': day_of_year,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'season': season,
            'holiday_name': holiday_name
        })
        
        current += timedelta(days=1)
    
    df_calendar = pd.DataFrame(calendar_data)
    df_calendar.to_csv('data/raw/calendar_data.csv', index=False)
    print(f"   ✅ Saved: calendar_data.csv ({len(df_calendar)} dates)")
    
    # 4. Generate Sales Data
    print("4️⃣  Generating sales data...")
    
    np.random.seed(42)
    sales_data = []
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    dates_list = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for store_id in range(1, 6):
        for item_id in range(101, 111):
            base_demand = np.random.uniform(20, 200)
            unit_cost = df_products[df_products['item_id'] == item_id]['unit_cost'].values[0]
            
            for day_idx, date in enumerate(dates_list):
                # Components
                trend = (day_idx / len(dates_list)) * 0.2 * base_demand
                
                day_of_week = date.weekday()
                weekly_pattern = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_week / 7)
                
                day_of_year = date.timetuple().tm_yday
                yearly_pattern = 1.0 + 0.4 * np.sin(2 * np.pi * day_of_year / 365)
                
                is_promo = np.random.random() < 0.15
                promo_effect = 2.0 if is_promo else 1.0
                discount = 0.3 if is_promo else 0.0
                
                base_price = unit_cost * np.random.uniform(2.0, 3.0)
                current_price = base_price * (1 - discount)
                
                noise = np.random.normal(1.0, 0.15)
                qty_sold = max(0, int((base_demand + trend) * weekly_pattern * yearly_pattern * promo_effect * noise))
                
                if item_id > 105:  # Last 5 items are intermittent
                    if np.random.random() > 0.85:
                        qty_sold = 0
                
                on_hand = np.random.randint(50, 500)
                stockout_flag = 0
                
                sales_data.append({
                    'store_id': store_id,
                    'item_id': item_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'qty_sold': qty_sold,
                    'price': round(current_price, 2),
                    'on_promo': int(is_promo),
                    'discount_pct': round(discount * 100, 1),
                    'on_hand': on_hand,
                    'stockout_flag': stockout_flag,
                    'unit_cost': unit_cost
                })
    
    df_sales = pd.DataFrame(sales_data)
    df_sales.to_csv('data/raw/retail_sales_data.csv', index=False)
    print(f"   ✅ Saved: retail_sales_data.csv ({len(df_sales)} records)")
    
    # 5. Copy to processed (same as raw initially)
    print("5️⃣  Creating processed dataset...")
    df_sales.to_csv('data/processed/processed_sales.csv', index=False)
    print(f"   ✅ Saved: processed_sales.csv ({len(df_sales)} records)")
    
    print(f"\n✅ ALL DATA GENERATED SUCCESSFULLY!")
    print(f"\nSummary:")
    print(f"  • Products: {len(df_products)}")
    print(f"  • Stores: {len(df_stores)}")
    print(f"  • Calendar Days: {len(df_calendar)}")
    print(f"  • Sales Records: {len(df_sales)}")
    print(f"  • Date Range: 2023-01-01 to 2024-12-31 ({len(df_calendar)} days)")
    print(f"  • Combinations: {len(df_stores)} stores × {len(df_products)} products = {len(df_stores) * len(df_products)} SKUs")

if __name__ == "__main__":
    generate_all_data()