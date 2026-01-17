import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_financials(ticker, sentiment_df, timeframe_days=30):
    if sentiment_df.empty:
        return sentiment_df
    
    try:
        start_date = sentiment_df.index.min()
        end_date = sentiment_df.index.max()
        
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(end_date, 'date'):
            end_date = end_date.date()
            
        end_date = end_date + timedelta(days=1)
        
        if timeframe_days <= 1:
            interval = "1h"
            period = "2d"
        else:
            interval = "1d"
            period = None
        
        stock = yf.Ticker(ticker)
        
        if period:
            try:
                price_data = stock.history(period=period, interval=interval)
                print(f"Fetched price data using period='{period}', interval='{interval}': {len(price_data)} records")
            except Exception as e:
                print(f"Period-based fetch failed: {e}, trying date range...")
                price_data = stock.history(start=start_date, end=end_date, interval="1d")
                interval = "1d"
        else:
            price_data = stock.history(start=start_date, end=end_date, interval=interval)
        
        if price_data.empty:
            print("No price data found for the given date range")
            return sentiment_df
        
        if timeframe_days <= 1 and period:
            start_timestamp = pd.Timestamp(start_date).tz_localize('UTC')
            if price_data.index.tz is not None:
                start_timestamp = start_timestamp.tz_convert(price_data.index.tz)
            price_data = price_data[price_data.index >= start_timestamp]
        
        if timeframe_days <= 1 and interval == "1h":
            price_resampled = price_data.resample('h').last()
        else:
            price_resampled = price_data.resample('D').last()
            price_resampled.index = price_resampled.index.date
        
        print(f"Original price data: {len(price_data)} records from {price_data.index.min()} to {price_data.index.max()}")
        print(f"Price data timezone: {price_data.index.tz}")
        print(f"Sentiment data timezone: {getattr(sentiment_df.index, 'tz', 'No timezone')}")
        print(f"Resampled price data: {len(price_resampled)} records")
        print(f"Sentiment data: {len(sentiment_df)} records from {sentiment_df.index.min()} to {sentiment_df.index.max()}")
        
        merged_df = sentiment_df.join(price_resampled[['Close', 'Volume']], how='left')
        
        print(f"Price data shape: {price_resampled.shape}")
        print(f"Sentiment data shape: {sentiment_df.shape}")
        print(f"Merged data shape: {merged_df.shape}")
        print(f"Close price NaN count: {merged_df['Close'].isna().sum()}")
        
        merged_df['Close'] = merged_df['Close'].ffill()
        merged_df['Volume'] = merged_df['Volume'].fillna(0)
        
        print(f"Successfully merged sentiment data with price data for {len(merged_df)} time periods")
        return merged_df
        
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return sentiment_df

def calculate_verdict(df):
    if df.empty:
        return None

    df = df.sort_index(ascending=True)
    
    n = len(df)
    weights = np.linspace(0.2, 1.0, n)
    
    weighted_sentiment = np.average(df['Compound_Score'], weights=weights)
    
    sentiment_score = (weighted_sentiment + 1) * 5
    
    value_score = 5.0
    price_change_pct = 0.0
    price_reasoning = "No price data available."
    
    if 'Close' in df.columns and not df['Close'].isna().all():
        first_price = df['Close'].dropna().iloc[0]
        last_price = df['Close'].dropna().iloc[-1]
        price_change_pct = ((last_price - first_price) / first_price) * 100
        
        if price_change_pct < -15:
            value_score = 9.0
            price_reasoning = f"Price has dropped {abs(price_change_pct):.2f}%, making it a STRONG BUY opportunity."
        elif price_change_pct < -5:
            value_score = 7.0
            price_reasoning = f"Price has declined {abs(price_change_pct):.2f}%, creating good VALUE."
        elif price_change_pct > 15:
            value_score = 2.0
            price_reasoning = f"Price has rallied {price_change_pct:.2f}%, making it VERY EXPENSIVE."
        elif price_change_pct > 5:
            value_score = 3.0
            price_reasoning = f"Price has rallied {price_change_pct:.2f}%, making it EXPENSIVE."
        elif -5 <= price_change_pct <= 5:
            value_score = 6.0
            price_reasoning = f"Price has moved {price_change_pct:+.2f}%, showing STABLE pricing."
        else:
            price_reasoning = f"Price has changed {price_change_pct:+.2f}%."

    if weighted_sentiment > 0.3:
        sentiment_reasoning = "News sentiment is VERY POSITIVE."
    elif weighted_sentiment > 0.1:
        sentiment_reasoning = "News sentiment is MILDLY POSITIVE."
    elif weighted_sentiment > -0.1:
        sentiment_reasoning = "News sentiment is NEUTRAL."
    elif weighted_sentiment > -0.3:
        sentiment_reasoning = "News sentiment is MILDLY NEGATIVE."
    else:
        sentiment_reasoning = "News sentiment is VERY NEGATIVE."

    final_score = (sentiment_score * 0.6) + (value_score * 0.4)
    
    explanation = f"{sentiment_reasoning} {price_reasoning}"
    
    return {
        'Sentiment_Score': round(sentiment_score, 1),
        'Value_Score': round(value_score, 1),
        'Final_Score': round(final_score, 1),
        'Explanation': explanation
    }