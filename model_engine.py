from prophet import Prophet
from sklearn.metrics import mean_absolute_error

def train_model(df):
    """Trains the Prophet model on the provided data."""
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    return model

def get_accuracy_score(df):
    """
    Splits data to verify accuracy (Train on 80%, Test on 20%).
    Returns the Mean Absolute Error (MAE).
    """
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    
    val_model = Prophet(daily_seasonality=True)
    val_model.fit(train_df)
    
    prediction = val_model.predict(test_df)
    mae = mean_absolute_error(test_df['y'], prediction['yhat'])
    
    return mae