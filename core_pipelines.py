import yfinance as yf
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time
from bs4 import BeautifulSoup
from typing import Tuple, List, Dict, Any

# Internal metadata asset index to give the UI an institutional look and feel
ASSET_METADATA = {
    "AAPL": {"name": "Apple Inc.", "exchange": "NASDAQ: AAPL", "sector": "Technology Sector"},
    "MSFT": {"name": "Microsoft Corporation", "exchange": "NASDAQ: MSFT", "sector": "Technology Sector"},
    "GOOGL": {"name": "Alphabet Inc.", "exchange": "NASDAQ: GOOGL", "sector": "Technology Sector"},
    "AMZN": {"name": "Amazon.com Inc.", "exchange": "NASDAQ: AMZN", "sector": "Consumer Cyclical"},
    "NVDA": {"name": "NVIDIA Corporation", "exchange": "NASDAQ: NVDA", "sector": "Technology Sector"},
    "TSLA": {"name": "Tesla Inc.", "exchange": "NASDAQ: TSLA", "sector": "Consumer Cyclical"},
    "GS": {"name": "The Goldman Sachs Group", "exchange": "NYSE: GS", "sector": "Financial Services"},
    "DB": {"name": "Deutsche Bank AG", "exchange": "NYSE: DB", "sector": "Financial Services"},
    "RELIANCE.NS": {"name": "Reliance Industries Ltd.", "exchange": "NSE: RELIANCE", "sector": "Energy & Conglomerate"},
    "TCS.NS": {"name": "Tata Consultancy Services", "exchange": "NSE: TCS", "sector": "Technology Services"},
    "INFY.NS": {"name": "Infosys Limited", "exchange": "NSE: INFY", "sector": "Technology Services"}
}

def get_asset_profile(ticker_str: str) -> Dict[str, str]:
    """
    Returns structured naming metadata for a ticker string if available.
    Falls back gracefully to generic tokens if the user searches a custom asset.
    """
    return ASSET_METADATA.get(ticker_str, {
        "name": f"{ticker_str} Asset", 
        "exchange": f"Global Market: {ticker_str}", 
        "sector": "General Security"
    })

def fetch_historical_data(ticker_symbol: str, period: str = "1y") -> Tuple[bool, pd.DataFrame]:
    """
    Fetches historical market data via yfinance.
    Implements validation guardrails to prevent down-stream interface breakages.
    """
    ticker_str = str(ticker_symbol).strip().upper()
    if not ticker_str:
        return False, pd.DataFrame()
        
    start_time = time.time()
    try:
        ticker = yf.Ticker(ticker_str)
        df = ticker.history(period=period)
        
        if df.empty:
            return False, pd.DataFrame()
            
        df = df.reset_index()
        df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
        
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return False, pd.DataFrame()
                
        df = df.ffill().bfill()
        print(f"[SUCCESS] Ingested {len(df)} rows for {ticker_str} in {time.time() - start_time:.4f}s")
        return True, df
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Data ingestion system failure: {str(e)}")
        return False, pd.DataFrame()

def _clean_html_text(raw_text: str) -> str:
    """
    Strips raw HTML fragments and text debris out of description payloads.
    """
    if not raw_text:
        return ""
    try:
        cleaned = BeautifulSoup(raw_text, "html.parser").get_text()
        cleaned = " ".join(cleaned.split())
        return cleaned
    except Exception:
        return raw_text

def fetch_ticker_news(ticker_symbol: str) -> List[Dict[str, Any]]:
    """
    Extracts asset news streams using a cascading yfinance -> RSS fallback pattern.
    """
    ticker_str = str(ticker_symbol).strip().upper()
    structured_news = []
    
    if not ticker_str:
        return structured_news

    try:
        ticker = yf.Ticker(ticker_str)
        raw_news = ticker.news
        
        if raw_news and len(raw_news) > 0:
            for item in raw_news:
                title = _clean_html_text(item.get("title", ""))
                summary = _clean_html_text(item.get("summary", item.get("description", "")))
                source = item.get("publisher", item.get("source", "Market Feed")).strip()
                
                if title:
                    structured_news.append({
                        "title": title,
                        "summary": summary if summary else title,
                        "source": source
                    })
            if structured_news:
                return structured_news
    except Exception as e:
        print(f"[NEWS WARNING] Primary pipeline throttled for {ticker_str}: {str(e)}")

    try:
        fallback_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_str}"
        headers = {"User-Agent": "AlphaIntelTerminal/1.0 (Contact: student-dev@iitr.ac.in)"}
        response = requests.get(fallback_url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.findall(".//item"):
                title = item.find("title")
                description = item.find("description")
                source = item.find("source")
                
                title_text = _clean_html_text(title.text) if title is not None else ""
                desc_text = _clean_html_text(description.text) if description is not None else ""
                source_text = source.text.strip() if source is not None else "Yahoo Finance"
                
                if title_text:
                    structured_news.append({
                        "title": title_text,
                        "summary": desc_text if desc_text else title_text,
                        "source": source_text
                    })
    except Exception as rss_err:
        print(f"[NEWS CRITICAL] Total news blackout: {str(rss_err)}")
        
    return structured_news