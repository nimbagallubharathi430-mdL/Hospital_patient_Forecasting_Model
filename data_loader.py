import pandas as pd
import numpy as np

def generate_data():
    """
    Simulates realistic hospital traffic data:
    1. Trend (Hospital growing)
    2. Weekly (Weekends are quiet)
    3. Yearly (Winter flu season)
    """

    dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
    n = len(dates)
    
  
    baseline = 50
    trend = np.linspace(0, 20, n)
    weekly = np.where(dates.dayofweek < 5, 15, -5) 
    yearly = 20 * np.cos((dates.dayofyear - 1) * 2 * np.pi / 365)
    noise = np.random.normal(0, 3, n)
    
    y = baseline + trend + weekly + yearly + noise
    df = pd.DataFrame({'ds': dates, 'y': np.maximum(y, 0)})
    
    return df