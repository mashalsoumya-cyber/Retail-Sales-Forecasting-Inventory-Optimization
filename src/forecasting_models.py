"""
Forecasting Models Module (RF, Croston, Seasonal Naive)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from src.utils import log_message

class RandomForestForecaster:
    """Random Forest Regressor for forecasting regular demand"""
    
    def __init__(self, n_estimators=400, max_depth=12, min_samples_leaf=5, random_state=13):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the model"""
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        log_message("Random Forest model trained")
    
    def predict(self, X_test):
        """Make predictions"""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def feature_importance(self):
        """Get feature importance"""
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importances

class CrostonForecaster:
    """
    Croston's method for intermittent demand forecasting
    
    Decompose demand into:
    - z_t: average demand size (given a sale occurs)
    - p_t: average period between sales
    
    Forecast = z / p
    """
    
    def __init__(self, alpha=0.1, apply_sba=True):
        """
        alpha: smoothing parameter (0-1)
        apply_sba: Apply Syntetos-Boylan Adjustment for bias correction
        """
        self.alpha = alpha
        self.apply_sba = apply_sba
        self.z_init = None
        self.p_init = None
    
    def fit(self, y):
        """
        Fit Croston's model to a demand series
        y: pandas Series of demand
        """
        demand = y.values
        non_zero_idx = np.where(demand > 0)[0]
        
        if len(non_zero_idx) == 0:
            self.z_init = 0
            self.p_init = 1
            return
        
        # Non-zero demand values
        z_values = demand[non_zero_idx]
        
        # Periods between sales
        periods = np.diff(np.r_[0, non_zero_idx])
        periods = periods[periods > 0]  # Exclude first zero
        
        self.z_init = z_values[0]
        self.p_init = periods[0] if len(periods) > 0 else 1
        
        # Smoothing
        z_hat = self.z_init
        p_hat = self.p_init
        
        for i in range(1, len(z_values)):
            z_hat = self.alpha * z_values[i] + (1 - self.alpha) * z_hat
        
        for i in range(1, len(periods)):
            p_hat = self.alpha * periods[i] + (1 - self.alpha) * p_hat
        
        self.z_init = z_hat
        self.p_init = p_hat
    
    def forecast(self, horizon=30):
        """
        Generate forecast for h periods ahead
        """
        if self.z_init is None or self.p_init is None:
            return np.zeros(horizon)
        
        # Base forecast
        f = (self.z_init / self.p_init) * np.ones(horizon)
        
        # SBA correction: multiply by (1 - alpha/2)
        if self.apply_sba:
            f = f * (1 - self.alpha / 2)
        
        return np.maximum(f, 0)  # Ensure non-negative

class SeasonalNaiveForecaster:
    """
    Seasonal Naive forecaster: use same day last year or last week
    """
    
    def __init__(self, season_length=7):
        """
        season_length: 7 for weekly seasonality, 365 for yearly
        """
        self.season_length = season_length
        self.y_train = None
    
    def fit(self, y):
        """Fit with training data"""
        self.y_train = y.values
    
    def forecast(self, horizon=30):
        """Forecast using seasonal pattern"""
        forecast = []
        for i in range(horizon):
            idx = len(self.y_train) - self.season_length + (i % self.season_length)
            if idx >= 0 and idx < len(self.y_train):
                forecast.append(self.y_train[idx])
            else:
                forecast.append(self.y_train[-1])  # fallback
        
        return np.array(forecast)

def select_forecaster(y, intermittency_threshold=0.3, feature_X=None):
    """
    Select appropriate forecaster based on demand pattern
    
    High intermittency (P0 > threshold) → Croston
    Low intermittency → Random Forest (if features available) or Seasonal Naive
    """
    
    p0 = (y == 0).mean()  # Intermittency metric
    
    if p0 > intermittency_threshold:
        # Intermittent demand → Croston
        return 'croston', p0
    else:
        # Regular demand → RF if features, else Seasonal Naive
        if feature_X is not None and len(feature_X) > 0:
            return 'random_forest', p0
        else:
            return 'seasonal_naive', p0

def ensemble_forecast(y_rf, y_croston, y_naive, weights=None):
    """
    Ensemble multiple forecasts with weighted average

    Default weights: RF=0.5, Croston=0.3, Naive=0.2
    """

    y_rf = np.asarray(y_rf)
    y_croston = np.asarray(y_croston)
    y_naive = np.asarray(y_naive)

    if weights is None:
        weights = np.array([0.5, 0.3, 0.2], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1 or weights.size != 3:
        raise ValueError("weights must be an iterable of three numeric values")

    weights = weights / weights.sum()  # Normalize

    return weights[0] * y_rf + weights[1] * y_croston + weights[2] * y_naive