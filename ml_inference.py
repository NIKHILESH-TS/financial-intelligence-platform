import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from typing import Tuple, Dict, Any

def generate_price_prediction(df: pd.DataFrame, lag_days: int = 5) -> Tuple[bool, Dict[str, Any]]:
    """
    Generates single-step forward price points with variance tracking intervals.
    Enforces restricted depth metrics to block localized historical memorization.
    """
    payload = {
        "predicted_price": 0.0,
        "model_accuracy_score": 0.0,
        "direction": "Undetermined",
        "confidence_low": 0.0,
        "confidence_high": 0.0,
        "feature_importances": {},
        "lag_days": lag_days
    }
    
    if df.empty or 'close' not in df.columns or len(df) < (lag_days * 3):
        if not df.empty and 'close' in df.columns:
            val = float(df['close'].iloc[-1])
            payload["predicted_price"] = val
            payload["confidence_low"] = val * 0.98
            payload["confidence_high"] = val * 1.02
        return False, payload
        
    try:
        working_df = df[['date', 'close']].copy()
        
        feature_cols = []
        for i in range(1, lag_days + 1):
            col_name = f'lag_{i}'
            working_df[col_name] = working_df['close'].shift(i)
            feature_cols.append(col_name)
            
        working_df['target'] = working_df['close'].shift(-1)
        clean_df = working_df.dropna()
        
        X = clean_df[feature_cols].values
        y = clean_df['target'].values
        
        # Human Design: Explicit depth controls to enforce structural regularization bounds
        model = RandomForestRegressor(
            n_estimators=50, 
            max_depth=4, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X, y)
        
        train_preds = model.predict(X)
        rmse = root_mean_squared_error(y, train_preds)
        
        payload["model_accuracy_score"] = max(0.0, float(model.score(X, y)))
        
        importances = model.feature_importances_
        payload["feature_importances"] = {
            f"Lag {i}": float(importances[i-1] * 100.0) for i, col in enumerate(feature_cols)
        }
        
        most_recent_features = working_df[feature_cols].iloc[-1].values.reshape(1, -1)
        
        if np.isnan(most_recent_features).any():
            val = float(df['close'].iloc[-1])
            payload["predicted_price"] = val
            return True, payload
            
        predicted_val = float(model.predict(most_recent_features)[0])
        current_close = float(df['close'].iloc[-1])
        
        payload["predicted_price"] = predicted_val
        payload["confidence_low"] = float(predicted_val - (1.96 * rmse))
        payload["confidence_high"] = float(predicted_val + (1.96 * rmse))
        
        if predicted_val > current_close:
            payload["direction"] = "Upward"
        elif predicted_val < current_close:
            payload["direction"] = "Downward"
        else:
            payload["direction"] = "Unchanged"
            
        return True, payload
        
    except Exception as e:
        print(f"[DEVELOPER NOTICE] Scikit-Learn training loop bypassed: {str(e)}")
        return False, payload