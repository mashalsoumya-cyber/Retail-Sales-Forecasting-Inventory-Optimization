"""
Exploratory Data Analysis Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import log_message
from src.config import OUTPUT_IMAGES_PATH

def calculate_intermittency_metric(df):
    """
    Calculate P0 (percentage of zero-demand periods) for each SKU
    Items with high P0 need Croston or Poisson models
    """
    
    intermittency = df.groupby(['store_id', 'item_id'])['qty_sold'].apply(
        lambda x: (x == 0).mean()
    ).reset_index()
    intermittency.columns = ['store_id', 'item_id', 'P0']
    
    log_message(f"Intermittency metric (P0) calculated")
    print(f"\nP0 Distribution:")
    print(f"  Mean P0: {intermittency['P0'].mean():.2%}")
    print(f"  Min P0: {intermittency['P0'].min():.2%}")
    print(f"  Max P0: {intermittency['P0'].max():.2%}")
    
    return intermittency

def analyze_trend_seasonality(df, skus_to_plot=3):
    """
    Analyze trends and seasonality in the data
    """
    
    fig, axes = plt.subplots(skus_to_plot, 1, figsize=(14, 10))
    
    # Sample random SKUs
    sku_samples = df.groupby(['store_id', 'item_id']).size().nlargest(skus_to_plot).index
    
    for idx, (store_id, item_id) in enumerate(sku_samples):
        data = df[(df['store_id'] == store_id) & (df['item_id'] == item_id)].sort_values('date')
        
        axes[idx].plot(data['date'], data['qty_sold'], marker='o', linewidth=2, markersize=4)
        axes[idx].set_title(f'Sales Trend - Store {store_id}, Item {item_id}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Date')
        axes[idx].set_ylabel('Quantity Sold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = f"{OUTPUT_IMAGES_PATH}eda_trends.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    log_message(f"Saved: {filepath}")
    plt.close()

def analyze_promo_impact(df):
    """
    Analyze impact of promotions on sales
    """
    
    if 'on_promo' not in df.columns:
        log_message("⚠️  'on_promo' column not found, skipping promo analysis")
        return None
    
    promo_impact = df.groupby('on_promo')['qty_sold'].agg(['count', 'mean', 'std']).round(2)
    promo_impact.index = ['No Promo', 'On Promo']
    
    print(f"\nPromotion Impact Analysis:")
    print(promo_impact)
    
    # Calculate lift
    no_promo_mean = df[df['on_promo'] == 0]['qty_sold'].mean()
    promo_mean = df[df['on_promo'] == 1]['qty_sold'].mean()
    lift = ((promo_mean - no_promo_mean) / no_promo_mean * 100) if no_promo_mean > 0 else 0
    
    print(f"\n  Promo Lift: {lift:.1f}%")
    
    return promo_impact

def analyze_category_performance(df, category_col=None):
    """
    Analyze performance by category (if category info exists)
    """
    
    if category_col is None or category_col not in df.columns:
        log_message("Category column not provided or not found, skipping category analysis")
        return None
    
    category_stats = df.groupby(category_col)['qty_sold'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('total', 'sum')
    ]).round(2).sort_values('total', ascending=False)
    
    print(f"\nCategory Performance:")
    print(category_stats)
    
    return category_stats

def generate_eda_summary(df):
    """
    Generate comprehensive EDA summary
    """
    
    log_message("Generating EDA Summary...")
    
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("="*70)
    
    # Basic Statistics
    print(f"\nDataset Size: {len(df)} rows × {len(df.columns)} columns")
    print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique Stores: {df['store_id'].nunique()}")
    print(f"Unique Products: {df['item_id'].nunique()}")
    
    # Sales Statistics
    print(f"\nSales Statistics (qty_sold):")
    print(f"  Mean: {df['qty_sold'].mean():.2f} units/day")
    print(f"  Median: {df['qty_sold'].median():.2f} units/day")
    print(f"  Std Dev: {df['qty_sold'].std():.2f} units/day")
    print(f"  Min: {df['qty_sold'].min():.0f} units/day")
    print(f"  Max: {df['qty_sold'].max():.0f} units/day")
    
    # Zero-demand analysis
    zero_pct = (df['qty_sold'] == 0).mean() * 100
    print(f"\nZero-Demand Days: {zero_pct:.1f}%")
    
    # Intermittency
    intermittency = calculate_intermittency_metric(df)
    
    # Promotion analysis (if available)
    if 'on_promo' in df.columns:
        analyze_promo_impact(df)
    
    # Generate visualizations
    analyze_trend_seasonality(df, skus_to_plot=3)
    
    print("\n" + "="*70 + "\n")
    
    return intermittency

def plot_sales_distribution(df, filepath=None):
    """Plot distribution of sales"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(df['qty_sold'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Quantity Sold')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Daily Sales')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(df['qty_sold'], vert=True)
    ax2.set_ylabel('Quantity Sold')
    ax2.set_title('Sales Distribution - Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        log_message(f"Saved: {filepath}")
    
    plt.close()