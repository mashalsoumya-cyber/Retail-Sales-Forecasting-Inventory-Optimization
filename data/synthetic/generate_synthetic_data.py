"""
Generate Synthetic Retail Sales Data for Testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_retail_data(
    n_days=730,  # 2 years of data
    n_stores=3,
    n_products=20,
    output_path="data/raw/retail_sales_data.csv"
):
    """
    Generate realistic synthetic retail sales data
    
    Includes:
    - Trend (slight upward over time)
    - Seasonality (weekly, yearly patterns)
    - Promotional impact
    - Store variations
    - Product variations
    """
    
    print(f"Generating synthetic retail data...")
    print(f"  Days: {n_days}")
    print(f"  Stores: {n_stores}")
    print(f"  Products: {n_products}")
    
    # Date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=int(d)) for d in range(n_days)]
    
    data = []
    
    np.random.seed(42)  # For reproducibility
    
    for store_id in range(1, n_stores + 1):
        for item_id in range(1, n_products + 1):
            
            # Base demand (varies by product and store)
            base_demand = np.random.uniform(20, 200)
            
            for day_idx, date in enumerate(dates):
                
                # 1. Trend component
                trend = (day_idx / n_days) * 0.2 * base_demand
                
                # 2. Weekly seasonality (higher sales on weekends)
                day_of_week = date.weekday()
                weekly_pattern = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_week / 7)
                
                # 3. Yearly seasonality
                day_of_year = date.timetuple().tm_yday
                yearly_pattern = 1.0 + 0.4 * np.sin(2 * np.pi * day_of_year / 365)
                
                # 4. Promotional effect (random promotions)
                promo_probability = 0.15  # 15% of days have promotions
                is_promo = np.random.random() < promo_probability
                promo_effect = 2.0 if is_promo else 1.0  # 2x sales during promo
                discount = 0.3 if is_promo else 0.0
                
                # 5. Base price (varies by product)
                base_price = np.random.uniform(100, 500)
                current_price = base_price * (1 - discount)
                
                # 6. Random noise
                noise = np.random.normal(1.0, 0.15)
                
                # Combine all components
                qty_sold = (base_demand + trend) * weekly_pattern * yearly_pattern * promo_effect * noise
                qty_sold = max(0, int(qty_sold))  # Ensure non-negative
                
                # Randomly create intermittent demand (some products have intermittent patterns)
                if item_id > n_products * 0.7:  # Last 30% are intermittent
                    if np.random.random() > 0.85:  # 85% zero sales for intermittent items
                        qty_sold = 0
                
                # Stock information
                on_hand = np.random.randint(50, 500)
                stockout_flag = 0  # No stockouts in synthetic data
                
                data.append({
                    'store_id': store_id,
                    'item_id': item_id,
                    'date': date,
                    'qty_sold': qty_sold,
                    'price': round(current_price, 2),
                    'on_promo': int(is_promo),
                    'discount_pct': round(discount * 100, 1),
                    'on_hand': on_hand,
                    'stockout_flag': stockout_flag,
                    'unit_cost': round(base_price * 0.4, 2)  # 40% margin
                })
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Synthetic data saved: {output_path}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Stores: {df['store_id'].nunique()}")
    print(f"   Products: {df['item_id'].nunique()}")
    
    return df

if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_retail_data(
        n_days=730,
        n_stores=3,
        n_products=20,
        output_path="../../data/raw/retail_sales_data.csv"
    )
    
    print("\nData Preview:")
    print(df.head(10))
    print(f"\nData Summary:")
    print(df.describe())