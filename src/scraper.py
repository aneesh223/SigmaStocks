import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import yfinance as yf
from gnews import GNews
from datetime import datetime, timedelta
import pandas as pd
import pytz

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

def scrape_finviz(ticker, days=30):
    profile_data = get_profile_data(ticker)
    
    print(f"Searching Google News (RSS) for {ticker} over last {days} days...")
    parsed = []
    
    cutoff_date = datetime.now(pytz.utc) - timedelta(days=days + 2)
    
    if days <= 7:
        g_period = '7d'
    elif days <= 31:
        g_period = '1m'
    else:
        g_period = '1y'

    try:
        google_news = GNews(language='en', country='US')
        
        search_queries = [
            f"{ticker} stock",
            f"{ticker}",
            f"{ticker} earnings",
            f"{ticker} news"
        ]
        
        json_resp = []
        for query in search_queries:
            try:
                results = google_news.get_news(query)
                if results:
                    json_resp.extend(results)
                    print(f"Query '{query}' returned {len(results)} articles")
                    break
            except Exception as e:
                print(f"Query '{query}' failed: {e}")
                continue
        
        print(f"GNews returned {len(json_resp)} total articles")
        
        for item in json_resp:
            title = item.get('title')
            date_str = item.get('published date')
            publisher = item.get('publisher', {}).get('title', 'Unknown')
            
            dt_obj = None
            
            if date_str:
                try:
                    dt_obj = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S GMT')
                    dt_obj = dt_obj.replace(tzinfo=pytz.utc)
                except ValueError:
                    try:
                        dt_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except ValueError:
                        dt_obj = None
            
            if dt_obj and dt_obj < cutoff_date:
                continue
            
            if dt_obj is None:
                dt_obj = datetime.now(pytz.utc)
            
            parsed.append([dt_obj, title, publisher])
            
        print(f"Found {len(parsed)} relevant articles.")
        
    except Exception as e:
        print(f"GNews Error: {e}")
        
    return profile_data, parsed