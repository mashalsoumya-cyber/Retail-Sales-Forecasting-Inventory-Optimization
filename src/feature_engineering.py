"""
Feature Engineering Module
"""

import pandas as pd
import numpy as np
from src.utils import log_message

def create_lag_features(df, lags=[1, 7, 14, 30]):
    """
    Create lagged features
    lag_1: sales from 1 day ago
    lag_7: sales from 7 days ago (same day last week)
    lag_14: sales from 14 days ago
    lag_30: sales from 30 days ago
    """
    
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['store_id', 'item_id'])['qty_sold'].shift(lag)
    
    log_message(f"Created lag features: {lags}")
    return df

def create_rolling_features(df, windows=[7, 14, 28]):
    """
    Create rolling window statistics
    rollmean_7: 7-day moving average
    rollstd_7: 7-day moving std dev
    rollmin_7: 7-day minimum
    rollmax_7: 7-day maximum
    """
    
    for window in windows:
        # Calculate rolling mean (using shifted values to avoid look-ahead bias)
        df[f'rollmean_{window}'] = df.groupby(['store_id', 'item_id'])['qty_sold'].shift(1).rolling(window=window).mean()
        
        # Calculate rolling std
        df[f'rollstd_{window}'] = df.groupby(['store_id', 'item_id'])['qty_sold'].shift(1).rolling(window=window).std()
        
        # Calculate rolling min/max
        df[f'rollmin_{window}'] = df.groupby(['store_id', 'item_id'])['qty_sold'].shift(1).rolling(window=window).min()
        df[f'rollmax_{window}'] = df.groupby(['store_id', 'item_id'])['qty_sold'].shift(1).rolling(window=window).max()
    
    log_message(f"Created rolling features: windows {windows}")
    return df

def create_calendar_features(df):
    """
    Create calendar-based features
    """
    
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    log_message("Created calendar features")
    return df

def create_promotional_features(df):
    """
    Create promotional and price-based features
    """
    
    if 'price' in df.columns:
        # Price deviation from average
        df['price_avg'] = df.groupby(['store_id', 'item_id'])['price'].transform('mean')
        df['price_deviation'] = (df['price'] - df['price_avg']) / df['price_avg']
        df['price_change'] = df.groupby(['store_id', 'item_id'])['price'].diff()
        
        log_message("Created price-based features")
    
    if 'on_promo' in df.columns:
        # Promo streak (consecutive days on promo)
        df['promo_streak'] = df.groupby(['store_id', 'item_id', 'on_promo']).cumcount() + 1
        df.loc[df['on_promo'] == 0, 'promo_streak'] = 0
        
        log_message("Created promotion features")
    
    return df

def create_target_encoding_features(df):
    """
    Create target-encoded features (mean encoding by category)
    """
    
    # Mean sales by day of week
    dow_mean = df.groupby('day_of_week')['qty_sold'].mean()
    df['dow_mean_qty'] = df['day_of_week'].map(dow_mean)
    
    # Mean sales by month
    month_mean = df.groupby('month')['qty_sold'].mean()
    df['month_mean_qty'] = df['month'].map(month_mean)
    
    log_message("Created target-encoded features")
    return df

def engineer_features(df, lags=[1, 7, 14, 30], rolling_windows=[7, 14, 28]):
    """
    Main feature engineering pipeline
    """
    
    log_message("Starting feature engineering pipeline...")
    
    df = df.copy()
    
    # Sort by store, item, date
    df = df.sort_values(['store_id', 'item_id', 'date'])
    
    # Create features
    df = create_lag_features(df, lags=lags)
    df = create_rolling_features(df, windows=rolling_windows)
    df = create_calendar_features(df)
    df = create_promotional_features(df)
    df = create_target_encoding_features(df)
    
    # Drop rows with NaN (from lags and rolling features)
    original_rows = len(df)
    df = df.dropna()
    dropped_rows = original_rows - len(df)
    
    log_message(f"Dropped {dropped_rows} rows with NaN values (cold start)")
    log_message(f"✅ Feature engineering completed: {len(df)} rows with {len(df.columns)} columns")
    
    return df

def get_feature_columns(df):
    """
    Get list of feature columns (exclude target and id columns)
    """
    
    exclude_cols = {'store_id', 'item_id', 'date', 'qty_sold', 'on_hand', 'stockout_flag'}
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    return feature_cols