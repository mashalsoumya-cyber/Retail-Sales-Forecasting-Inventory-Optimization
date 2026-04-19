"""
Visualization Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import log_message
from src.config import OUTPUT_IMAGES_PATH, COLOR_ACTUAL, COLOR_FORECAST, COLOR_UPPER_CI, COLOR_LOWER_CI

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_forecast_vs_actual(dates, actual, forecast, lower_ci=None, upper_ci=None, 
                             title="Sales Forecast", filename=None):
    """
    Plot actual sales vs forecasted sales with confidence intervals
    """
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Actual sales
    ax.plot(dates[:len(actual)], actual, label='Actual', color=COLOR_ACTUAL, linewidth=2.5, marker='o', markersize=5)
    
    # Forecasted sales
    ax.plot(dates[len(actual):len(actual)+len(forecast)], forecast, label='Forecast', 
            color=COLOR_FORECAST, linewidth=2.5, marker='s', markersize=5)
    
    # Confidence intervals
    if lower_ci is not None and upper_ci is not None:
        forecast_dates = dates[len(actual):len(actual)+len(forecast)]
        ax.fill_between(forecast_dates, lower_ci, upper_ci, alpha=0.2, color=COLOR_FORECAST, label='95% CI')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quantity Sold', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"{OUTPUT_IMAGES_PATH}{filename}", dpi=300, bbox_inches='tight')
        log_message(f"Saved: {OUTPUT_IMAGES_PATH}{filename}")
    
    plt.close()

def plot_inventory_heatmap(recommendations_df, filename=None):
    """
    Plot heatmap of replenishment priorities
    """
    
    if len(recommendations_df) == 0:
        log_message("No recommendations to plot")
        return
    
    # Create priority matrix
    pivot_data = recommendations_df.pivot_table(
        values='priority',
        index='store_id',
        columns='item_id',
        aggfunc='first'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(pivot_data, annot=True, fmt='d', cmap='RdYlGn_r', cbar_kws={'label': 'Priority'},
                ax=ax, linewidths=0.5)
    
    ax.set_title('Replenishment Priority Heatmap\n(1=URGENT, 2=SOON, 3=STABLE)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Product ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Store ID', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"{OUTPUT_IMAGES_PATH}{filename}", dpi=300, bbox_inches='tight')
        log_message(f"Saved: {OUTPUT_IMAGES_PATH}{filename}")
    
    plt.close()

def plot_model_metrics(metrics_dict, filename=None):
    """Plot model performance metrics as bars"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize metrics for visualization
    metrics_to_plot = {k: v for k, v in metrics_dict.items() if k in ['MAE', 'RMSE', 'MAPE']}
    
    bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Error Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"{OUTPUT_IMAGES_PATH}{filename}", dpi=300, bbox_inches='tight')
        log_message(f"Saved: {OUTPUT_IMAGES_PATH}{filename}")
    
    plt.close()

def plot_category_performance(df, category_col, filename=None):
    """Plot sales performance by category"""
    
    cat_stats = df.groupby(category_col)['qty_sold'].agg(['mean', 'sum']).sort_values('sum', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average sales by category
    cat_stats['mean'].plot(kind='barh', ax=ax1, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Average Daily Sales', fontsize=11, fontweight='bold')
    ax1.set_title(f'Average Sales by {category_col}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Total sales by category
    cat_stats['sum'].plot(kind='barh', ax=ax2, color='coral', edgecolor='black')
    ax2.set_xlabel('Total Sales', fontsize=11, fontweight='bold')
    ax2.set_title(f'Total Sales by {category_col}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"{OUTPUT_IMAGES_PATH}{filename}", dpi=300, bbox_inches='tight')
        log_message(f"Saved: {OUTPUT_IMAGES_PATH}{filename}")
    
    plt.close()