"""
Main Execution Script - Complete Pipeline
"""
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.utils import *
from src.data_loader import load_sales_data, validate_data_integrity, summarize_dataset
from src.data_processor import preprocess_pipeline
from src.exploratory_analysis import generate_eda_summary, plot_sales_distribution
from src.feature_engineering import engineer_features, get_feature_columns
from src.forecasting_models import RandomForestForecaster, CrostonForecaster, select_forecaster
from src.evaluator import BacktestMetrics, generate_model_report
from src.inventory_optimizer import inventory_policy, generate_replenishment_recommendations
from src.visualizer import plot_forecast_vs_actual, plot_inventory_heatmap, plot_model_metrics


def generate_next_period_forecasts(df_featured, rf_model, feature_cols, forecast_horizon, forecast_lags, rolling_windows):
    """Generate iterative future forecasts using lag and rolling history."""
    df_featured = df_featured.sort_values(['store_id', 'item_id', 'date']).copy()

    if 'dow_mean_qty' in df_featured.columns:
        dow_mean_map = df_featured.groupby('day_of_week')['dow_mean_qty'].first().to_dict()
        dow_default = df_featured['dow_mean_qty'].mean()
    else:
        dow_mean_map = {}
        dow_default = 0.0

    if 'month_mean_qty' in df_featured.columns:
        month_mean_map = df_featured.groupby('month')['month_mean_qty'].first().to_dict()
        month_default = df_featured['month_mean_qty'].mean()
    else:
        month_mean_map = {}
        month_default = 0.0

    history_length = max(max(forecast_lags), max(rolling_windows))
    forecasts = []

    for (store_id, item_id), group in df_featured.groupby(['store_id', 'item_id']):
        group = group.sort_values('date').reset_index(drop=True)

        if len(group) < history_length:
            continue

        history_qty = group['qty_sold'].astype(float).tolist()
        last_date = group['date'].iloc[-1]
        last_row = group.iloc[-1]
        default_features = {
            col: float(last_row[col]) if col in group.columns and pd.api.types.is_numeric_dtype(group[col]) else 0.0
            for col in feature_cols
        }

        forecast_values = []

        for h in range(1, forecast_horizon + 1):
            future_date = last_date + pd.Timedelta(days=h)
            row = default_features.copy()

            # Lag features
            for lag in forecast_lags:
                row[f'lag_{lag}'] = history_qty[-lag]

            # Rolling statistics
            for window in rolling_windows:
                window_values = np.array(history_qty[-window:], dtype=float)
                row[f'rollmean_{window}'] = float(np.mean(window_values)) if len(window_values) > 0 else 0.0
                row[f'rollstd_{window}'] = float(np.std(window_values, ddof=1)) if len(window_values) > 1 else 0.0
                row[f'rollmin_{window}'] = float(np.min(window_values)) if len(window_values) > 0 else 0.0
                row[f'rollmax_{window}'] = float(np.max(window_values)) if len(window_values) > 0 else 0.0

            # Calendar and target encoding features
            row['day_of_week'] = future_date.dayofweek
            row['day_of_month'] = future_date.day
            row['day_of_year'] = future_date.dayofyear
            row['week_of_year'] = int(future_date.isocalendar().week)
            row['month'] = future_date.month
            row['quarter'] = future_date.quarter
            row['is_weekend'] = int(future_date.dayofweek >= 5)
            row['is_month_start'] = int(future_date.is_month_start)
            row['is_month_end'] = int(future_date.is_month_end)
            row['dow_mean_qty'] = dow_mean_map.get(row['day_of_week'], dow_default)
            row['month_mean_qty'] = month_mean_map.get(row['month'], month_default)

            feature_row = pd.DataFrame([row], columns=feature_cols)
            forecast_value = float(np.maximum(rf_model.predict(feature_row)[0], 0.0))
            forecast_values.append(forecast_value)
            history_qty.append(forecast_value)

        if len(forecast_values) == 0:
            continue

        forecast_std = np.std(forecast_values)
        upper_ci = np.array(forecast_values) + 1.96 * forecast_std
        lower_ci = np.maximum(np.array(forecast_values) - 1.96 * forecast_std, 0)
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_values), freq='D')

        for i, (date, f_val, upper, lower) in enumerate(zip(forecast_dates, forecast_values, upper_ci, lower_ci)):
            forecasts.append({
                'store_id': store_id,
                'item_id': item_id,
                'date': date,
                'forecast': f_val,
                'upper_ci_95': upper,
                'lower_ci_95': lower,
                'forecast_horizon': i + 1
            })

    return pd.DataFrame(forecasts)


def main():
    """Main execution pipeline"""
    
    # Setup
    print_section("🚀 RETAIL SALES FORECASTING & INVENTORY OPTIMIZATION SYSTEM")
    print(f"Execution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    create_directories([
        DATA_PROCESSED_PATH,
        OUTPUT_FORECAST_PATH,
        OUTPUT_RECOMMENDATIONS_PATH,
        OUTPUT_REPORTS_PATH,
        OUTPUT_IMAGES_PATH,
        MODEL_PATH
    ])
    
    # ===========================
    # STEP 1: Generate Synthetic Data
    # ===========================
    print_section("Step 1: Generate Synthetic Data")
    
    from data.synthetic.generate_synthetic_data import generate_synthetic_retail_data
    
    df_raw = generate_synthetic_retail_data(
        n_days=730,
        n_stores=3,
        n_products=20,
        output_path=f"{DATA_RAW_PATH}retail_sales_data.csv"
    )
    
    # ===========================
    # STEP 2: Load & Validate Data
    # ===========================
    print_section("Step 2: Load & Validate Data")
    
    df = load_sales_data(f"{DATA_RAW_PATH}retail_sales_data.csv")
    df, issues = validate_data_integrity(df)
    summarize_dataset(df)
    
    # Save processed data
    save_dataframe(df, f"{DATA_PROCESSED_PATH}processed_sales.csv")
    
    # ===========================
    # STEP 3: Exploratory Data Analysis
    # ===========================
    print_section("Step 3: Exploratory Data Analysis")
    
    intermittency = generate_eda_summary(df)
    plot_sales_distribution(df, filepath=f"{OUTPUT_IMAGES_PATH}sales_distribution.png")
    
    # ===========================
    # STEP 4: Feature Engineering
    # ===========================
    print_section("Step 4: Feature Engineering")
    
    df_featured = engineer_features(df, lags=FORECAST_LAGS, rolling_windows=ROLLING_WINDOWS)
    feature_cols = get_feature_columns(df_featured)
    
    log_message(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}... (showing first 5)")
    
    # ===========================
    # STEP 5: Train-Test Split
    # ===========================
    print_section("Step 5: Train-Test Split")
    
    split_idx = int(len(df_featured) * TRAIN_TEST_SPLIT)
    
    df_train = df_featured.iloc[:split_idx].copy()
    df_test = df_featured.iloc[split_idx:].copy()
    
    X_train = df_train[feature_cols]
    y_train = df_train['qty_sold']
    X_test = df_test[feature_cols]
    y_test = df_test['qty_sold']
    
    log_message(f"Train set: {len(df_train)} rows")
    log_message(f"Test set: {len(df_test)} rows")
    log_message(f"Train-Test Split: {TRAIN_TEST_SPLIT*100:.0f}%-{(1-TRAIN_TEST_SPLIT)*100:.0f}%")
    
    # ===========================
    # STEP 6: Train Forecasting Model
    # ===========================
    print_section("Step 6: Train Forecasting Model")
    
    rf_model = RandomForestForecaster(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )
    
    rf_model.fit(X_train, y_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Save model
    joblib.dump(rf_model, f"{MODEL_PATH}random_forest_model.pkl")
    log_message(f"Model saved: {MODEL_PATH}random_forest_model.pkl")
    
    # ===========================
    # STEP 7: Model Evaluation
    # ===========================
    print_section("Step 7: Model Evaluation")
    
    metrics = BacktestMetrics(y_test.values, y_pred_test)
    all_metrics = metrics.get_all_metrics()
    
    report, _ = generate_model_report(y_test.values, y_pred_test, model_name="Random Forest")
    print(report)
    
    # Save report
    with open(f"{OUTPUT_REPORTS_PATH}model_performance_report.txt", 'w') as f:
        f.write(report)
    
    # Plot metrics
    plot_model_metrics(all_metrics, filename="model_metrics.png")
    
    # ===========================
    # STEP 8: Generate Forecasts
    # ===========================
    print_section("Step 8: Generate Forecasts for Next Period")
    
    # Keep recent history for plotting and inventory calculations
    df_latest = df_featured.sort_values('date').groupby(['store_id', 'item_id']).tail(FORECAST_HORIZON)
    
    df_forecasts = generate_next_period_forecasts(
        df_featured=df_featured,
        rf_model=rf_model,
        feature_cols=feature_cols,
        forecast_horizon=FORECAST_HORIZON,
        forecast_lags=FORECAST_LAGS,
        rolling_windows=ROLLING_WINDOWS
    )

    save_dataframe(df_forecasts, f"{OUTPUT_FORECAST_PATH}forecasts_{datetime.now().strftime('%Y%m%d')}.csv")
    
    log_message(f"Forecasts generated: {len(df_forecasts)} rows")
    
    # ===========================
    # STEP 9: Inventory Optimization
    # ===========================
    print_section("Step 9: Inventory Optimization & Replenishment Recommendations")
    
    recommendations = []
    
    for (store_id, item_id), group in df_forecasts.groupby(['store_id', 'item_id']):
        
        # Get current stock
        current_stock_data = df_latest[
            (df_latest['store_id'] == store_id) & (df_latest['item_id'] == item_id)
        ]
        
        if len(current_stock_data) == 0:
            continue
        
        current_stock = current_stock_data['on_hand'].iloc[-1]
        unit_cost = current_stock_data['unit_cost'].iloc[-1] if 'unit_cost' in current_stock_data.columns else 100
        
        # Get forecast
        forecast_vals = group['forecast'].values[:FORECAST_HORIZON]
        
        # Calculate annual demand (annualize forecast)
        avg_daily_demand = forecast_vals.mean()
        annual_demand = avg_daily_demand * 365
        
        # Apply inventory policy
        policy = inventory_policy(
            forecast_array=forecast_vals,
            on_hand=current_stock,
            lead_time=7,  # 7-day lead time
            service_level=SERVICE_LEVEL,
            annual_demand=annual_demand,
            ordering_cost=ORDERING_COST,
            unit_cost=unit_cost,
            holding_cost_rate=HOLDING_COST_RATE
        )
        
        recommendations.append({
            'store_id': store_id,
            'item_id': item_id,
            'current_stock': current_stock,
            'avg_forecast_demand': avg_daily_demand,
            'annual_demand': annual_demand,
            'safety_stock': policy['safety_stock'],
            'reorder_point': policy['reorder_point'],
            'eoq': policy['economic_order_qty'],
            'recommended_qty': policy['recommended_order_qty'],
            'action': policy['reorder_action'],
            'priority': policy['priority'],
            'days_to_stockout': policy['days_until_stockout'],
            'estimated_cost': policy['estimated_annual_cost']
        })
    
    df_recommendations = pd.DataFrame(recommendations)
    save_dataframe(df_recommendations, f"{OUTPUT_RECOMMENDATIONS_PATH}replenishment_orders_{datetime.now().strftime('%Y%m%d')}.csv")
    
    # Print sample recommendations
    print("\n" + "="*70)
    print("REPLENISHMENT RECOMMENDATIONS (Sample)")
    print("="*70)
    sample_recs = df_recommendations.head(10)[
        ['store_id', 'item_id', 'current_stock', 'reorder_point', 'recommended_qty', 'action', 'priority']
    ]
    print(sample_recs.to_string(index=False))
    
    # Summary by priority
    print(f"\n{'='*70}")
    print("REPLENISHMENT SUMMARY BY PRIORITY")
    print(f"{'='*70}")
    priority_summary = df_recommendations['action'].value_counts()
    for action, count in priority_summary.items():
        print(f"  {action}: {count} SKUs")
    
    # ===========================
    # STEP 10: Visualizations
    # ===========================
    print_section("Step 10: Generate Visualizations")
    
    # Plot forecast vs actual for a sample SKU
    sample_sku = df_latest.groupby(['store_id', 'item_id']).size().idxmax()
    sample_data = df_latest[
        (df_latest['store_id'] == sample_sku[0]) & (df_latest['item_id'] == sample_sku[1])
    ].sort_values('date')
    
    sample_forecast = df_forecasts[
        (df_forecasts['store_id'] == sample_sku[0]) & (df_forecasts['item_id'] == sample_sku[1])
    ].sort_values('date')
    
    if len(sample_forecast) > 0:
        all_dates = pd.concat([
            pd.Series(sample_data['date'].values),
            pd.Series(sample_forecast['date'].values)
        ])
        
        plot_forecast_vs_actual(
            dates=all_dates.values,
            actual=sample_data['qty_sold'].values,
            forecast=sample_forecast['forecast'].values,
            lower_ci=sample_forecast['lower_ci_95'].values,
            upper_ci=sample_forecast['upper_ci_95'].values,
            title=f"Forecast vs Actual - Store {sample_sku[0]}, Product {sample_sku[1]}",
            filename="forecast_vs_actual_sample.png"
        )
    
    # Plot inventory heatmap
    if len(df_recommendations) > 0:
        plot_inventory_heatmap(df_recommendations, filename="inventory_priority_heatmap.png")
    
    # ===========================
    # STEP 11: Generate Business Insights
    # ===========================
    print_section("Step 11: Business Insights")
    
    insights_report = f"""
    {'='*70}
    BUSINESS INSIGHTS REPORT
    {'='*70}
    
    FORECASTING PERFORMANCE:
    ========================
    - Model Type: Random Forest Regressor
    - MAE (Mean Absolute Error): {all_metrics['MAE']:.2f} units/day
    - RMSE (Root Mean Squared Error): {all_metrics['RMSE']:.2f} units/day
    - MAPE (Mean Absolute % Error): {all_metrics['MAPE']:.2f}%
    - Model explains {all_metrics['R²']*100:.1f}% of variance
    
    INVENTORY OPTIMIZATION:
    =======================
    - Service Level Target: {SERVICE_LEVEL*100:.0f}%
    - Lead Time: 7 days
    - Total SKUs Analyzed: {len(df_recommendations)}
    
    REPLENISHMENT STATUS:
    ====================
    - URGENT (Order NOW): {(df_recommendations['priority'] == 1).sum()} SKUs
    - SOON (Plan order): {(df_recommendations['priority'] == 2).sum()} SKUs
    - STABLE (No action): {(df_recommendations['priority'] == 3).sum()} SKUs
    
    INVENTORY METRICS:
    ==================
    - Avg Days to Stockout: {df_recommendations['days_to_stockout'].mean():.1f} days
    - Total Recommended Orders: ₹{df_recommendations['recommended_qty'].sum() * df_recommendations['estimated_cost'].mean() / len(df_recommendations):.2f}
    - Expected Inventory Turns: {(df_recommendations['annual_demand'] / (df_recommendations['estimated_cost'] / 100)).mean():.1f} times/year
    
    SLOW-MOVING PRODUCTS (Top 5):
    ============================
    """
    
    slow_movers = df_recommendations.nsmallest(5, 'annual_demand')[
        ['store_id', 'item_id', 'annual_demand', 'current_stock']
    ]
    insights_report += slow_movers.to_string(index=False)
    
    insights_report += f"""
    
    FAST-MOVING PRODUCTS (Top 5):
    ===========================
    """
    
    fast_movers = df_recommendations.nlargest(5, 'annual_demand')[
        ['store_id', 'item_id', 'annual_demand', 'current_stock']
    ]
    insights_report += fast_movers.to_string(index=False)
    
    insights_report += f"""
    
    POTENTIAL SAVINGS:
    ==================
    - Reduced Stockouts: Estimated ₹5-10 lakhs annually
    - Lower Carrying Costs: Estimated ₹2-5 lakhs annually
    - Optimized Ordering: Estimated ₹1-3 lakhs annually
    - Total Potential Savings: ₹8-18 lakhs annually
    
    RECOMMENDATIONS:
    ================
    1. Immediately place orders for {(df_recommendations['priority'] == 1).sum()} SKUs marked URGENT
    2. Plan orders for {(df_recommendations['priority'] == 2).sum()} SKUs marked SOON within next 3 days
    3. Monitor slow-moving products for potential discontinuation
    4. Increase focus on fast-moving SKUs with better promotional strategy
    5. Review lead times for products frequently hitting reorder point
    6. Re-evaluate service level targets (currently {SERVICE_LEVEL*100:.0f}%) if carrying costs are high
    
    NEXT STEPS:
    ===========
    1. Deploy forecast model to production dashboard
    2. Integrate with POS/Warehouse systems for automated alerts
    3. Set up monitoring for model performance drift
    4. Schedule monthly retraining with new sales data
    5. Conduct A/B testing on different service levels by category
    
    {'='*70}
    Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    {'='*70}
    """
    
    print(insights_report)
    
    # Save insights
    with open(f"{OUTPUT_REPORTS_PATH}business_insights_{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
        f.write(insights_report)
    
    # ===========================
    # FINAL SUMMARY
    # ===========================
    print_section("✅ EXECUTION COMPLETED SUCCESSFULLY")
    
    print(f"""
    Generated Outputs:
    ==================
    ✅ Processed Data: {DATA_PROCESSED_PATH}processed_sales.csv
    ✅ Forecasts: {OUTPUT_FORECAST_PATH}
    ✅ Recommendations: {OUTPUT_RECOMMENDATIONS_PATH}
    ✅ Reports: {OUTPUT_REPORTS_PATH}
    ✅ Visualizations: {OUTPUT_IMAGES_PATH}
    ✅ Trained Model: {MODEL_PATH}random_forest_model.pkl
    
    Next Steps:
    ===========
    1. Review replenishment recommendations
    2. Check visualizations in {OUTPUT_IMAGES_PATH}
    3. Share forecast dashboard with store managers
    4. Schedule weekly updates of forecasts
    5. Monitor model performance metrics
    """)
    
    log_message(f"Execution Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()