"""
Configuration file for Retail Sales Forecasting & Inventory Optimization System
"""

# Data Configuration
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
DATA_SYNTHETIC_PATH = "data/synthetic/"

# Model Configuration
MODEL_PATH = "models/"
FORECAST_HORIZON = 30  # Days to forecast
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

# Inventory Parameters
SERVICE_LEVEL = 0.95  # 95% service level
HOLDING_COST_RATE = 0.25  # 25% of unit cost per year
ORDERING_COST = 500  # ₹ per order

# Forecasting Parameters
FORECAST_LAGS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 28]
INTERMITTENCY_THRESHOLD = 0.3  # P0 threshold for Croston

# Model Parameters
RF_N_ESTIMATORS = 400
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 5
RANDOM_STATE = 13

# Croston Parameters
CROSTON_ALPHA = 0.1

# Visualization Colors
COLOR_ACTUAL = '#1f77b4'
COLOR_FORECAST = '#ff7f0e'
COLOR_UPPER_CI = '#d62728'
COLOR_LOWER_CI = '#2ca02c'

# Output Paths
OUTPUT_FORECAST_PATH = "outputs/forecasts/"
OUTPUT_RECOMMENDATIONS_PATH = "outputs/recommendations/"
OUTPUT_REPORTS_PATH = "outputs/reports/"
OUTPUT_IMAGES_PATH = "images/"

# Logging
LOG_FILE = "outputs/reports/execution_log.txt"