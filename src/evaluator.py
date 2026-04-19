"""
Model Evaluation & Backtesting Module
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from src.utils import log_message, calculate_mase

class BacktestMetrics:
    """Calculate backtesting metrics"""
    
    def __init__(self, y_true, y_pred, y_naive=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_naive = y_naive if y_naive is not None else np.mean(y_true) * np.ones_like(y_true)
    
    def mae(self):
        """Mean Absolute Error"""
        return mean_absolute_error(self.y_true, self.y_pred)
    
    def rmse(self):
        """Root Mean Squared Error"""
        mse = mean_squared_error(self.y_true, self.y_pred)
        return np.sqrt(mse)
    
    def mape(self):
        """Mean Absolute Percentage Error"""
        mask = self.y_true != 0
        if mask.sum() == 0:
            return np.nan
        return mean_absolute_percentage_error(self.y_true[mask], self.y_pred[mask])
    
    def mase(self):
        """Mean Absolute Scaled Error"""
        numerator = mean_absolute_error(self.y_true, self.y_pred)
        denominator = mean_absolute_error(self.y_true[1:], self.y_true[:-1])
        
        if denominator == 0:
            return np.nan
        return numerator / denominator
    
    def r_squared(self):
        """R-Squared / Coefficient of Determination"""
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        
        if ss_tot == 0:
            return np.nan
        return 1 - (ss_res / ss_tot)
    
    def get_all_metrics(self):
        """Get dictionary of all metrics"""
        return {
            'MAE': self.mae(),
            'RMSE': self.rmse(),
            'MAPE': self.mape(),
            'MASE': self.mase(),
            'R²': self.r_squared()
        }

def rolling_origin_backtest(df, model, feature_cols, window_size=30, step_size=7):
    """
    Rolling origin cross-validation for time series
    
    window_size: initial training window
    step_size: step forward for each iteration
    """
    
    predictions = []
    actuals = []
    
    n = len(df)
    
    for i in range(window_size, n - step_size, step_size):
        # Training set
        train_df = df.iloc[:i]
        
        # Test set
        test_df = df.iloc[i:i+step_size]
        
        X_train = train_df[feature_cols]
        y_train = train_df['qty_sold']
        
        X_test = test_df[feature_cols]
        y_test = test_df['qty_sold']
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        predictions.extend(y_pred)
        actuals.extend(y_test.values)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return actuals, predictions

def evaluate_model_by_category(df_results, category_col):
    """
    Evaluate model performance by category
    
    df_results: DataFrame with columns ['actual', 'predicted', category_col]
    category_col: column name for grouping
    """
    
    results_by_cat = []
    
    for cat, group in df_results.groupby(category_col):
        metrics = BacktestMetrics(group['actual'].values, group['predicted'].values)
        all_metrics = metrics.get_all_metrics()
        all_metrics[category_col] = cat
        results_by_cat.append(all_metrics)
    
    results_df = pd.DataFrame(results_by_cat)
    return results_df

def generate_model_report(y_true, y_pred, model_name="Model"):
    """Generate a text report of model performance"""
    
    metrics = BacktestMetrics(y_true, y_pred)
    all_metrics = metrics.get_all_metrics()
    
    report = f"""
    {'='*70}
    MODEL EVALUATION REPORT: {model_name}
    {'='*70}
    
    Sample Size: {len(y_true)} predictions
    
    Metrics:
    --------
    MAE (Mean Absolute Error):           {all_metrics['MAE']:.4f} units
    RMSE (Root Mean Squared Error):      {all_metrics['RMSE']:.4f} units
    MAPE (Mean Absolute % Error):        {all_metrics['MAPE']:.2f}%
    MASE (Mean Absolute Scaled Error):   {all_metrics['MASE']:.4f}
    R² (Coefficient of Determination):   {all_metrics['R²']:.4f}
    
    Interpretation:
    ---------------
    - MAE: On average, predictions are off by {all_metrics['MAE']:.2f} units
    - RMSE: Penalizes larger errors more than MAE
    - MAPE: Relative error as percentage
    - MASE: Scaled error (< 1 is better than seasonal naive)
    - R²: Proportion of variance explained (0-1, higher is better)
    
    {'='*70}
    """
    
    return report, all_metrics