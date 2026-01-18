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
import pytz
from functools import lru_cache
import concurrent.futures
from threading import Lock
import time

# Cache for profile data to avoid repeated API calls
_profile_cache = {}
_cache_lock = Lock()

@lru_cache(maxsize=100)
def get_profile_data(ticker):
    """Get profile data with caching and error handling"""
    cache_key = ticker.upper()
    
    # Thread-safe cache check
    with _cache_lock:
        if cache_key in _profile_cache:
            cached_data, cache_time = _profile_cache[cache_key]
            # Cache for 1 hour
            if time.time() - cache_time < 3600:
                return cached_data
    
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
        
        result = (name, industry, mkt_cap)
        
        # Thread-safe cache update
        with _cache_lock:
            _profile_cache[cache_key] = (result, time.time())
        
        return result
        
    except Exception as e:
        print(f"API Error: {e}")
        fallback = (ticker.upper(), "N/A", "N/A")
        
        # Cache the fallback to avoid repeated failures
        with _cache_lock:
            _profile_cache[cache_key] = (fallback, time.time())
        
        return fallback

def _parse_gnews_date(date_str):
    """Optimized date parsing with multiple format support"""
    if not date_str:
        return datetime.now(pytz.utc)
    
    # Try common formats in order of likelihood
    formats = [
        '%a, %d %b %Y %H:%M:%S GMT',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            if fmt == '%a, %d %b %Y %H:%M:%S GMT':
                dt_obj = datetime.strptime(date_str, fmt)
                return dt_obj.replace(tzinfo=pytz.utc)
            elif 'Z' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                dt_obj = datetime.strptime(date_str, fmt)
                return pytz.utc.localize(dt_obj)
        except ValueError:
            continue
    
    # Fallback to current time
    return datetime.now(pytz.utc)

def _fetch_gnews_query(args):
    """Helper function for parallel GNews queries"""
    google_news, query, ticker = args
    
    try:
        results = google_news.get_news(query)
        if results:
            print(f"Query '{query}' returned {len(results)} articles")
            return results
        return []
    except Exception as e:
        print(f"Query '{query}' failed: {e}")
        return []

def scrape_gnews(ticker, days=30):
    """Optimized GNews scraping with parallel processing and caching"""
    profile_data = get_profile_data(ticker)
    
    print(f"Searching Google News (RSS) for {ticker} over last {days} days...")
    parsed = []
    
    # Pre-calculate cutoff date
    cutoff_date = datetime.now(pytz.utc) - timedelta(days=days + 2)

    try:
        # Optimized search queries - reduced from 4 to 3 most effective ones
        search_queries = [
            f"{ticker} stock",
            f"{ticker} earnings", 
            f"{ticker} news"
        ]
        
        # Use ThreadPoolExecutor for parallel API calls
        google_news = GNews(language='en', country='US')
        
        json_resp = []
        
        # Parallel execution of queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            query_args = [(google_news, query, ticker) for query in search_queries]
            future_to_query = {executor.submit(_fetch_gnews_query, args): args[1] for args in query_args}
            
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result(timeout=10)  # 10 second timeout per query
                    if results:
                        json_resp.extend(results)
                        # Break after first successful query to avoid redundancy
                        if len(json_resp) >= 50:  # Reasonable limit
                            break
                except concurrent.futures.TimeoutError:
                    print(f"Query '{query}' timed out")
                except Exception as e:
                    print(f"Query '{query}' failed with exception: {e}")
        
        print(f"GNews returned {len(json_resp)} total articles")
        
        # Vectorized article processing
        for item in json_resp:
            title = item.get('title')
            if not title:
                continue
                
            date_str = item.get('published date')
            publisher = item.get('publisher', {}).get('title', 'Unknown')
            
            # Optimized date parsing
            dt_obj = _parse_gnews_date(date_str)
            
            # Skip articles outside date range
            if dt_obj < cutoff_date:
                continue
            
            parsed.append([dt_obj, title, publisher])
        
        print(f"Found {len(parsed)} relevant articles after filtering.")
        
    except Exception as e:
        print(f"GNews Error: {e}")
        
    return profile_data, parsed