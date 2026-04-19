"""
Utility functions for the forecasting system
"""

import io
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.config import LOG_FILE

def setup_logging():
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    stream_handler = logging.StreamHandler(
        stream=io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    )
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, stream_handler]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def safe_print(message):
    """Print to console safely on non-UTF-8 terminals."""
    if not isinstance(message, str):
        message = str(message)
    try:
        print(message)
    except UnicodeEncodeError:
        encoded = message.encode(sys.stdout.encoding or 'utf-8', errors='replace')
        print(encoded.decode(sys.stdout.encoding or 'utf-8', errors='replace'))

def log_message(message, level="info"):
    """Log message to file and console"""
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    safe_print(f"[{level.upper()}] {message}")

def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title.upper()}")
    print("="*70 + "\n")

def check_file_exists(filepath):
    """Check if file exists"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return True

def create_directories(paths):
    """Create multiple directories if they don't exist"""
    for path in paths:
        os.makedirs(path, exist_ok=True)
    log_message(f"Directories created/verified: {len(paths)} folders")

def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_divide(numerator, denominator, fill_value=0):
    """Safe division without divide-by-zero errors"""
    return np.where(denominator == 0, fill_value, numerator / denominator)

def remove_outliers_iqr(data, column, multiplier=1.5):
    """Remove outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    original_len = len(data)
    data_cleaned = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)].copy()
    removed = original_len - len(data_cleaned)
    
    log_message(f"Removed {removed} outliers from {column} using IQR method")
    return data_cleaned

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf

def calculate_mase(y_true, y_pred, y_naive):
    """Calculate Mean Absolute Scaled Error"""
    numerator = np.mean(np.abs(y_true - y_pred))
    denominator = np.mean(np.abs(y_true - y_naive))
    return numerator / denominator if denominator > 0 else np.inf

def save_dataframe(df, filepath, index=False):
    """Save DataFrame to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index)
    log_message(f"Saved: {filepath} ({len(df)} rows)")

def load_dataframe(filepath):
    """Load DataFrame from CSV"""
    check_file_exists(filepath)
    df = pd.read_csv(filepath)
    log_message(f"Loaded: {filepath} ({len(df)} rows, {len(df.columns)} columns)")
    return df