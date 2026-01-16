import yfinance as yf
import pandas as pd
from datetime import timedelta

def get_financials(ticker, sentiment_df):
    data_start = sentiment_df.index.min()
    data_end = sentiment_df.index.max() + timedelta(days=1)
    
    print(f"Downloading prices from {data_start} to {data_end}...")
    stock_data = yf.download(ticker, start=data_start, end=data_end, progress=False)
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    if stock_data.index.tz is not None:
        stock_data.index = stock_data.index.tz_localize(None)

    stock_data['Pct_Change'] = stock_data['Close'].pct_change()

    merged = pd.merge(sentiment_df, stock_data[['Close', 'Pct_Change']], left_index=True, right_index=True, how='inner')
    
    merged['Buy_Score'] = merged['Compound_Score'] - merged['Pct_Change']
    
    return merged