"""
Data Preprocessing & Cleaning Module
"""

import pandas as pd
import numpy as np
from src.utils import log_message

def handle_missing_values(df, method='forward_fill'):
    """
    Handle missing values in the dataset
    
    Methods:
    - 'forward_fill': Forward fill method
    - 'interpolate': Linear interpolation
    - 'mean': Fill with mean value
    """
    
    missing_before = df.isnull().sum().sum()
    
    if method == 'forward_fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        df = df.fillna(method='bfill').fillna(method='ffill')
    elif method == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    missing_after = df.isnull().sum().sum()
    log_message(f"Missing values handled: {missing_before} → {missing_after}")
    
    return df

def remove_outliers_zscore(df, column, threshold=3):
    """Remove outliers using z-score method"""
    
    original_len = len(df)
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    df = df[z_scores < threshold].copy()
    removed = original_len - len(df)
    
    log_message(f"Removed {removed} outliers from {column} (z-score > {threshold})")
    return df

def normalize_numeric_columns(df, columns):
    """Normalize numeric columns to [0,1] range"""
    
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
    
    log_message(f"Normalized columns: {columns}")
    return df

def aggregate_to_weekly(df):
    """Aggregate daily data to weekly"""
    
    df = df.copy()
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='d')
    
    agg_dict = {'qty_sold': 'sum', 'price': 'mean', 'on_promo': 'max'}
    df_weekly = df.groupby(['store_id', 'item_id', 'week_start']).agg(agg_dict).reset_index()
    df_weekly = df_weekly.rename(columns={'week_start': 'date'})
    
    log_message(f"Aggregated data to weekly: {len(df)} daily rows → {len(df_weekly)} weekly rows")
    return df_weekly

def create_complete_date_range(df):
    """Fill missing dates with zero sales to ensure complete time series"""
    
    df = df.copy()
    
    # Create complete date range for each store-item combination
    complete_df = []
    
    for (store_id, item_id), group in df.groupby(['store_id', 'item_id']):
        date_range = pd.date_range(start=group['date'].min(), end=group['date'].max(), freq='D')
        temp_df = pd.DataFrame({'date': date_range, 'store_id': store_id, 'item_id': item_id})
        temp_df = temp_df.merge(group, on=['store_id', 'item_id', 'date'], how='left')
        temp_df['qty_sold'] = temp_df['qty_sold'].fillna(0)
        complete_df.append(temp_df)
    
    df_complete = pd.concat(complete_df, ignore_index=True).sort_values(['store_id', 'item_id', 'date'])
    
    log_message(f"Complete date range created: {len(df)} → {len(df_complete)} rows")
    return df_complete

def preprocess_pipeline(df, fill_missing_method='forward_fill', fill_dates=True, 
                       remove_outliers=False, normalize=False):
    """
    Complete preprocessing pipeline
    """
    
    log_message("Starting preprocessing pipeline...")
    
    # Step 1: Handle missing values
    df = handle_missing_values(df, method=fill_missing_method)
    
    # Step 2: Fill missing dates (optional)
    if fill_dates:
        df = create_complete_date_range(df)
    
    # Step 3: Remove outliers (optional)
    if remove_outliers:
        df = remove_outliers_zscore(df, 'qty_sold', threshold=3)
    
    # Step 4: Final validation
    assert df.isnull().sum().sum() == 0, "NaN values still present after preprocessing"
    
    log_message("✅ Preprocessing pipeline completed")
    
    return df