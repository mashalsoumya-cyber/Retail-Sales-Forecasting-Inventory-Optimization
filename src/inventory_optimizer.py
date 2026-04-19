"""
Inventory Optimization Module
Calculates Safety Stock, Reorder Point, EOQ, and Order Quantity
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from src.utils import log_message

def calculate_safety_stock(forecast_std, service_level=0.95, lead_time=7):
    """
    Calculate Safety Stock
    
    SS = z * σ_L
    where z = norm.ppf(service_level)
    σ_L = forecast_std * sqrt(lead_time)
    """
    
    z = norm.ppf(service_level)
    sigma_L = forecast_std * np.sqrt(lead_time)
    SS = z * sigma_L
    
    return SS

def calculate_reorder_point(forecast_mean, forecast_std, service_level=0.95, lead_time=7):
    """
    Calculate Reorder Point
    
    ROP = μ_L + SS
    where μ_L = mean demand during lead time
    SS = safety stock
    """
    
    SS = calculate_safety_stock(forecast_std, service_level, lead_time)
    ROP = forecast_mean + SS
    
    return ROP, SS

def calculate_eoq(annual_demand, ordering_cost, unit_cost, holding_cost_rate=0.25):
    """
    Calculate Economic Order Quantity
    
    EOQ = sqrt((2 * D * K) / H)
    where D = annual demand
    K = ordering cost per order
    H = holding cost = unit_cost * holding_cost_rate
    """
    
    H = unit_cost * holding_cost_rate
    
    if H <= 0 or annual_demand <= 0:
        return 0
    
    EOQ = np.sqrt((2 * annual_demand * ordering_cost) / H)
    
    return EOQ

def calculate_order_quantity(on_hand, rop, eoq):
    """
    Calculate recommended order quantity
    
    Q = max(0, max(EOQ, ROP - on_hand))
    """
    
    Q = max(0, max(eoq, rop - on_hand))
    return Q

def get_reorder_action(on_hand, rop, ss):
    """
    Determine replenishment action
    
    URGENT: on_hand < ROP (order immediately)
    SOON: ROP <= on_hand < ROP + SS/2 (plan order)
    STABLE: on_hand >= ROP + SS/2 (no action)
    """
    
    if on_hand < rop:
        action = "URGENT - Order NOW"
        priority = 1
    elif on_hand < rop + ss/2:
        action = "SOON - Plan order"
        priority = 2
    else:
        action = "STABLE - No action"
        priority = 3
    
    return action, priority

def inventory_policy(forecast_array, on_hand, lead_time=7, service_level=0.95,
                     annual_demand=None, ordering_cost=500, unit_cost=100,
                     holding_cost_rate=0.25):
    """
    Calculate complete inventory policy for one SKU
    
    forecast_array: numpy array of forecasted demand for next periods
    on_hand: current stock
    lead_time: procurement lead time in days
    service_level: target service level (0.95 = 95%)
    annual_demand: if None, estimated from forecast
    ordering_cost: cost per order
    unit_cost: cost per unit
    holding_cost_rate: annual holding cost as % of unit cost
    
    Returns: dict with all policy parameters
    """
    
    # Estimate mean demand during lead time
    mu_L = forecast_array[:lead_time].sum()
    
    # Estimate std of demand during lead time
    sigma_L = np.std(forecast_array[:lead_time]) * np.sqrt(lead_time) if len(forecast_array) > 0 else 1
    
    # Calculate Safety Stock
    z = norm.ppf(service_level)
    SS = z * sigma_L
    
    # Calculate Reorder Point
    ROP = mu_L + SS
    
    # Estimate annual demand if not provided
    if annual_demand is None:
        annual_demand = forecast_array.mean() * 365
    
    # Calculate EOQ
    H = unit_cost * holding_cost_rate
    if H > 0:
        EOQ = np.sqrt((2 * annual_demand * ordering_cost) / H)
    else:
        EOQ = mu_L
    
    # Calculate order quantity
    Q_order = calculate_order_quantity(on_hand, ROP, EOQ)
    
    # Determine action
    action, priority = get_reorder_action(on_hand, ROP, SS)
    
    # Estimate days until stockout (if no replenishment)
    if forecast_array.mean() > 0:
        days_until_stockout = on_hand / forecast_array.mean()
    else:
        days_until_stockout = np.inf
    
    policy = {
        'demand_during_lead_time': mu_L,
        'std_demand_lead_time': sigma_L,
        'z_score': z,
        'safety_stock': SS,
        'reorder_point': ROP,
        'economic_order_qty': EOQ,
        'recommended_order_qty': Q_order,
        'reorder_action': action,
        'priority': priority,
        'days_until_stockout': days_until_stockout,
        'estimated_annual_demand': annual_demand,
        'estimated_annual_cost': (annual_demand * unit_cost) + (Q_order/2 * unit_cost * holding_cost_rate)
    }
    
    return policy

def generate_replenishment_recommendations(df, forecasts_df, inventory_params):
    """
    Generate replenishment recommendations for all SKUs
    
    df: master data with on_hand and other details
    forecasts_df: forecasted demand
    inventory_params: dict with lead_time, service_level, etc.
    
    Returns: DataFrame with recommendations
    """
    
    recommendations = []
    
    for (store_id, item_id), group in forecasts_df.groupby(['store_id', 'item_id']):
        
        # Get current stock
        current_stock = df[(df['store_id'] == store_id) & (df['item_id'] == item_id)]['on_hand'].iloc[-1]
        
        # Get forecast
        forecast_values = group['forecast'].values
        
        # Apply inventory policy
        policy = inventory_policy(
            forecast_array=forecast_values,
            on_hand=current_stock,
            lead_time=inventory_params.get('lead_time', 7),
            service_level=inventory_params.get('service_level', 0.95),
            annual_demand=inventory_params.get('annual_demand'),
            ordering_cost=inventory_params.get('ordering_cost', 500),
            unit_cost=inventory_params.get('unit_cost', 100),
            holding_cost_rate=inventory_params.get('holding_cost_rate', 0.25)
        )
        
        recommendations.append({
            'store_id': store_id,
            'item_id': item_id,
            'current_stock': current_stock,
            'avg_forecast': forecast_values.mean(),
            'safety_stock': policy['safety_stock'],
            'reorder_point': policy['reorder_point'],
            'eoq': policy['economic_order_qty'],
            'recommended_order_qty': policy['recommended_order_qty'],
            'action': policy['reorder_action'],
            'priority': policy['priority'],
            'days_to_stockout': policy['days_until_stockout'],
            'estimated_annual_cost': policy['estimated_annual_cost']
        })
    
    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df