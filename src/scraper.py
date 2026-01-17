import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import yfinance as yf
from gnews import GNews
from datetime import datetime
import pandas as pd

def get_profile_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName', ticker.upper())
        industry = info.get('industry', 'N/A')
        
        raw_cap = info.get('marketCap', None)
        if raw_cap:
            if raw_cap >= 1_000_000_000_000:
                mkt_cap = f"${raw_cap / 1_000_000_000_000:.2f}T"
            elif raw_cap >= 1_000_000_000:
                mkt_cap = f"${raw_cap / 1_000_000_000:.2f}B"
            else:
                mkt_cap = f"${raw_cap / 1_000_000:.2f}M"
        else:
            mkt_cap = "N/A"
            
        return name, industry, mkt_cap
    except Exception as e:
        print(f"API Error: {e}")
        return ticker.upper(), "N/A", "N/A"

def scrape_finviz(ticker):
    profile_data = get_profile_data(ticker)
    
    print(f"Searching Google News (RSS) for {ticker}...")
    parsed = []
    
    try:
        google_news = GNews(language='en', country='US', period='30d')
        json_resp = google_news.get_news(f"{ticker} stock")
        
        for item in json_resp:
            title = item.get('title')
            date_str = item.get('published date')
            
            # --- NEW: EXTRACT PUBLISHER ---
            # GNews stores publisher in a dict: {'href': '...', 'title': 'Bloomberg'}
            publisher = item.get('publisher', {}).get('title', 'Unknown')
            # ------------------------------

            dt_obj = datetime.now()
            if date_str:
                try:
                    dt_obj = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S GMT')
                except ValueError:
                    pass
            
            # Append 3 items now: Date, Title, Publisher
            parsed.append([dt_obj, title, publisher])
            
        print(f"Found {len(parsed)} articles via GNews.")
        
    except Exception as e:
        print(f"GNews Error: {e}")
        
    return profile_data, parsed