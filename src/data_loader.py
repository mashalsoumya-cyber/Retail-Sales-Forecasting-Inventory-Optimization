"""
Data Loading & Validation Module
"""

import pandas as pd
import numpy as np
from src.utils import log_message, check_file_exists, load_dataframe
from src.config import DATA_RAW_PATH

def load_sales_data(filepath):
    """
    Load sales data and perform basic validation
    
    Expected columns: store_id, item_id, date, qty_sold, price, on_promo
    """
    df = load_dataframe(filepath)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Validate required columns
    required_cols = {'store_id', 'item_id', 'date', 'qty_sold'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")
    
    log_message(f"✅ Sales data validation passed")
    return df

def validate_data_integrity(df):
    """Perform comprehensive data validation"""
    
    issues = {}
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['store_id', 'item_id', 'date']).sum()
    if duplicates > 0:
        issues['duplicates'] = duplicates
        df = df.drop_duplicates(subset=['store_id', 'item_id', 'date'], keep='first')
        log_message(f"⚠️  Removed {duplicates} duplicate rows")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        issues['missing_values'] = missing[missing > 0].to_dict()
        log_message(f"⚠️  Missing values detected:\n{missing[missing > 0]}")
    
    # Check for negative quantities
    if (df['qty_sold'] < 0).any():
        neg_count = (df['qty_sold'] < 0).sum()
        issues['negative_qty'] = neg_count
        df = df[df['qty_sold'] >= 0].copy()
        log_message(f"⚠️  Removed {neg_count} rows with negative quantities")
    
    # Check for stockout-censored days
    if 'stockout_flag' in df.columns:
        stockouts = (df['stockout_flag'] == 1).sum()
        if stockouts > 0:
            log_message(f"⚠️  Found {stockouts} stockout-flagged records")
            # Optionally remove them for clean demand signal
            df = df[df['stockout_flag'] == 0].copy()
            log_message(f"    Removed stockout-flagged records for clean demand")
    
    log_message(f"✅ Data integrity validation completed")
    log_message(f"   Final dataset: {len(df)} rows × {len(df.columns)} columns")
    
    return df, issues

def summarize_dataset(df):
    """Print dataset summary statistics"""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nDate Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\nUnique Stores: {df['store_id'].nunique()}")
    print(f"Unique Products: {df['item_id'].nunique()}")
    
    print(f"\n{'qty_sold Statistics':}")
    print(f"  Mean: {df['qty_sold'].mean():.2f} units/day")
    print(f"  Median: {df['qty_sold'].median():.2f} units/day")
    print(f"  Std Dev: {df['qty_sold'].std():.2f} units/day")
    print(f"  Min: {df['qty_sold'].min():.2f} units/day")
    print(f"  Max: {df['qty_sold'].max():.2f} units/day")
    
    if 'price' in df.columns:
        print(f"\nPrice Statistics:")
        print(f"  Mean: ₹{df['price'].mean():.2f}")
        print(f"  Min: ₹{df['price'].min():.2f}")
        print(f"  Max: ₹{df['price'].max():.2f}")
    
    print("\n" + "="*70 + "\n")

def get_sample_data_info():
    """Print info about expected dataset structure"""
    info = """
    EXPECTED DATA STRUCTURE
    ========================
    
    Column Names (Minimum Required):
    - store_id (int): Unique store identifier
    - item_id (int): Unique product identifier
    - date (datetime): Transaction date
    - qty_sold (int): Quantity sold on that day
    
    Optional Columns:
    - price (float): Price of product on that date
    - on_promo (int): 0/1 flag for promotional day
    - discount_pct (float): Discount percentage
    - on_hand (int): Stock available
    - stockout_flag (int): 1 if stockout occurred, 0 otherwise
    
    Example CSV:
    store_id,item_id,date,qty_sold,price,on_promo
    1,101,2024-01-01,50,299.99,0
    1,101,2024-01-02,55,299.99,0
    1,101,2024-01-03,120,199.99,1
    ...
    """
    print(info)