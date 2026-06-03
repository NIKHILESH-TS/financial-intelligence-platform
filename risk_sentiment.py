import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, Any, List

def calculate_advanced_risk(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes annualized historical volatility, max drawdown, Sharpe ratio,
    and technical momentum signals (RSI, MACD) via vectorized operations.
    """
    metrics = {
        "annualized_volatility": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "recent_close": 0.0,
        "sharpe_ratio": 0.0,
        "rsi_14": 50.0,
        "macd_value": 0.0,
        "macd_signal": 0.0
    }
    
    if df.empty or 'close' not in df.columns or len(df) < 2:
        return metrics
        
    try:
        df = df.copy()
        df['daily_return'] = df['close'].pct_change()
        
        # --- CORE QUANT METRICS ---
        initial_price = float(df['close'].iloc[0])
        final_price = float(df['close'].iloc[-1])
        if initial_price == 0:
            return metrics
            
        metrics["total_return"] = ((final_price - initial_price) / initial_price) * 100.0
        metrics["recent_close"] = final_price
        
        # Volatility & Sharpe Ratio calculations
        daily_std = df['daily_return'].std(ddof=1)
        if not np.isnan(daily_std) and daily_std > 0:
            an_vol = float(daily_std * np.sqrt(252) * 100.0)
            metrics["annualized_volatility"] = an_vol
            # Standard 5% risk-free benchmark proxy
            annualized_return = metrics["total_return"] / (len(df) / 252.0)
            metrics["sharpe_ratio"] = float((annualized_return - 5.0) / an_vol)
            
        # Cumulative Peak-to-Trough Maximum Drawdown
        rolling_max = df['close'].cummax()
        drawdowns = np.where(rolling_max > 0, (df['close'] - rolling_max) / rolling_max, 0.0)
        metrics["max_drawdown"] = float(np.min(drawdowns) * 100.0)
        
        # --- TECHNICAL INDICATORS HAND-CODED VIA PANDAS ---
        # 1. 14-Day RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        
        # Apply Wilder's smoothing baseline mechanics
        if len(df) >= 15:
            rs = avg_gain.iloc[-1] / (avg_loss.iloc[-1] + 1e-9)
            metrics["rsi_14"] = float(100 - (100 / (1 + rs)))
            
        # 2. MACD (12, 26, 9 configuration array standard)
        if len(df) >= 26:
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            metrics["macd_value"] = float(macd_line.iloc[-1])
            metrics["macd_signal"] = float(signal_line.iloc[-1])
            
        return metrics
        
    except Exception as e:
        print(f"[DEVELOPER NOTICE] Analytical computation loop variance: {str(e)}")
        return metrics

def analyze_sentiment_pool(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates market language structures via VADER compound scoring logic.
    """
    summary = {"average_compound": 0.0, "sentiment_label": "Neutral", "article_count": len(news_list)}
    if not news_list:
        return summary
        
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for article in news_list:
            text = article.get("summary", article.get("title", ""))
            if text:
                scores.append(analyzer.polarity_scores(text).get("compound", 0.0))
                
        if scores:
            mean_score = float(np.mean(scores))
            summary["average_compound"] = mean_score
            if mean_score >= 0.05:
                summary["sentiment_label"] = "Bullish"
            elif mean_score <= -0.05:
                summary["sentiment_label"] = "Bearish"
        return summary
    except Exception as e:
        print(f"[DEVELOPER NOTICE] Sentiment worker boundary skipped: {str(e)}")
        return summary