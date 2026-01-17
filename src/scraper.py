import requests
from bs4 import BeautifulSoup
import ssl
from datetime import datetime, timedelta
import pytz
import yfinance as yf  # <--- NEW: Using API instead of scraping

# --- SSL BYPASS ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def format_market_cap(value):
    """
    Helper to turn 80000000000 into '80B'
    """
    if not value or value is None:
        return "N/A"
    
    try:
        val = float(value)
    except (ValueError, TypeError):
        return str(value)

    if val >= 1_000_000_000_000:
        return f"${val / 1_000_000_000_000:.2f}T"
    elif val >= 1_000_000_000:
        return f"${val / 1_000_000_000:.2f}B"
    elif val >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    else:
        return f"${val:.2f}"

def get_profile_data(ticker):
    """
    Fetches reliable profile data using yfinance API.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. Company Name
        name = info.get('longName', ticker.upper())
        
        # 2. Industry
        industry = info.get('industry', 'N/A')
        
        # 3. Market Cap (Formatted)
        raw_cap = info.get('marketCap', None)
        market_cap = format_market_cap(raw_cap)
        
        return name, industry, market_cap
        
    except Exception as e:
        print(f"API Error: {e}")
        return ticker.upper(), "N/A", "N/A"

def scrape_finviz(ticker):
    """
    Orchestrator: Gets Profile from API, News from FinViz
    """
    # 1. Get Profile Data (Reliable API)
    profile_data = get_profile_data(ticker)
    
    # 2. Get News Data (FinViz Scraper)
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    parsed = []
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find(id='news-table')
            
            est = pytz.timezone('US/Eastern')
            utc = pytz.utc
            
            current_date_str = None
            
            if table:
                for row in table.find_all('tr'):
                    if row.a is None:
                        continue
                    
                    headline = row.a.get_text()
                    date_scrape = row.td.text.split()

                    if len(date_scrape) == 1:
                        time_str = date_scrape[0]
                        date_str = current_date_str 
                    else:
                        date_str = date_scrape[0]
                        time_str = date_scrape[1]
                        current_date_str = date_str

                    if date_str and time_str:
                        if date_str == 'Today':
                            date_str = datetime.now(est).strftime('%b-%d-%y')
                        elif date_str == 'Yesterday':
                            yesterday = datetime.now(est) - timedelta(days=1)
                            date_str = yesterday.strftime('%b-%d-%y')

                        full_time_str = f"{date_str} {time_str}"
                        
                        try:
                            dt_obj = datetime.strptime(full_time_str, '%b-%d-%y %I:%M%p')
                            dt_est = est.localize(dt_obj)
                            dt_utc = dt_est.astimezone(utc)
                            parsed.append([dt_utc, headline])
                        except ValueError:
                            pass
                            
    except Exception as e:
        print(f"Scraping Error: {e}")

    # Return Tuple: ((Name, Ind, Cap), NewsList)
    return profile_data, parsed