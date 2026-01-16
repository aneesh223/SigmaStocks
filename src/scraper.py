import requests
from bs4 import BeautifulSoup
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def scrape_finviz(ticker):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to connect. Error: {response.status_code}")
        return None

    print("Data Retrieved... Parsing...")
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find(id='news-table')
    
    parsed = []
    current_date = None

    for row in table.find_all('tr'):
        if row.a is None:
            continue
        headline = row.a.get_text()
        date_scrape = row.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]
            date = current_date
        else:
            date = date_scrape[0]
            time = date_scrape[1]
            current_date = date
        
        parsed.append([date, time, headline])
        
    return parsed