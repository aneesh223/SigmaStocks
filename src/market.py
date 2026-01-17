import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_financials(ticker, sentiment_df, timeframe_days=30):
    """
    Combines the Sentiment Data with Price Data.
    """
    if sentiment_df.empty:
        return sentiment_df
    
    try:
        # Get stock price data for the same date range as sentiment data
        start_date = sentiment_df.index.min()
        end_date = sentiment_df.index.max()
        
        # Handle different index types (datetime vs date)
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(end_date, 'date'):
            end_date = end_date.date()
            
        end_date = end_date + timedelta(days=1)  # Add 1 day to include end date
        
        # Determine the appropriate interval for price data
        # yfinance limitations: hourly data only available for last 730 days and max 60 days per request
        if timeframe_days <= 1:
            interval = "1h"  # Hourly data for 1 day
            period = "2d"    # Get 2 days to ensure we have data
        else:
            interval = "1d"  # Daily data for 5 days and longer
            period = None    # Use date range instead
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        
        # Try with period first (more reliable for short timeframes)
        if period:
            try:
                price_data = stock.history(period=period, interval=interval)
                print(f"Fetched price data using period='{period}', interval='{interval}': {len(price_data)} records")
            except Exception as e:
                print(f"Period-based fetch failed: {e}, trying date range...")
                price_data = stock.history(start=start_date, end=end_date, interval="1d")
                interval = "1d"  # Fallback to daily
        else:
            price_data = stock.history(start=start_date, end=end_date, interval=interval)
        
        if price_data.empty:
            print("No price data found for the given date range")
            return sentiment_df
        
        # Filter price data to our actual date range if needed
        if timeframe_days <= 1 and period:
            # For 1-day timeframe using period, filter to our sentiment date range
            # Convert start_date to timezone-aware timestamp to match price data
            start_timestamp = pd.Timestamp(start_date).tz_localize('UTC')
            if price_data.index.tz is not None:
                start_timestamp = start_timestamp.tz_convert(price_data.index.tz)
            price_data = price_data[price_data.index >= start_timestamp]
        
        # Resample price data to match sentiment grouping
        if timeframe_days <= 1 and interval == "1h":
            # Hourly grouping - no resampling needed
            price_resampled = price_data.resample('h').last()
        else:
            # Daily grouping for 5 days and longer
            price_resampled = price_data.resample('D').last()
            price_resampled.index = price_resampled.index.date
        
        # Debug info
        print(f"Original price data: {len(price_data)} records from {price_data.index.min()} to {price_data.index.max()}")
        print(f"Price data timezone: {price_data.index.tz}")
        print(f"Sentiment data timezone: {getattr(sentiment_df.index, 'tz', 'No timezone')}")
        print(f"Resampled price data: {len(price_resampled)} records")
        print(f"Sentiment data: {len(sentiment_df)} records from {sentiment_df.index.min()} to {sentiment_df.index.max()}")
        
        # Debug info
        print(f"Original price data: {len(price_data)} records from {price_data.index.min()} to {price_data.index.max()}")
        print(f"Resampled price data: {len(price_resampled)} records")
        print(f"Sentiment data: {len(sentiment_df)} records from {sentiment_df.index.min()} to {sentiment_df.index.max()}")
        
        # Merge sentiment and price data
        merged_df = sentiment_df.join(price_resampled[['Close', 'Volume']], how='left')
        
        # Debug info
        print(f"Price data shape: {price_resampled.shape}")
        print(f"Sentiment data shape: {sentiment_df.shape}")
        print(f"Merged data shape: {merged_df.shape}")
        print(f"Close price NaN count: {merged_df['Close'].isna().sum()}")
        
        # Forward fill missing price data (for weekends/holidays)
        merged_df['Close'] = merged_df['Close'].ffill()
        merged_df['Volume'] = merged_df['Volume'].fillna(0)
        
        print(f"Successfully merged sentiment data with price data for {len(merged_df)} time periods")
        return merged_df
        
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return sentiment_df

def calculate_verdict(df):
    """
    Calculates the final Buy/Sell score using TIME DECAY.
    Newer news has a heavier impact on the score.
    """
    if df.empty:
        return None

    # 1. Sort by Date (Oldest -> Newest)
    df = df.sort_index(ascending=True)
    
    # 2. Generate Weights (Linear Decay)
    # Example: If we have 10 days of data:
    # Weights will look like: [0.1, 0.2, 0.3 ... 0.9, 1.0]
    n = len(df)
    weights = np.linspace(0.2, 1.0, n) # Starts at 0.2 (Oldest), Ends at 1.0 (Newest)
    
    # 3. Calculate Weighted Sentiment Score
    # Formula: Sum(Score * Weight) / Sum(Weights)
    weighted_sentiment = np.average(df['Compound_Score'], weights=weights)
    
    # Scale to 0-10 (Sentiment is -1.0 to 1.0)
    # -1.0 -> 0
    #  0.0 -> 5
    # +1.0 -> 10
    sentiment_score = (weighted_sentiment + 1) * 5
    
    # 4. Calculate Price/Value Score with detailed analysis
    value_score = 5.0 # Neutral placeholder if no price data
    price_change_pct = 0.0
    price_reasoning = "No price data available."
    
    if 'Close' in df.columns and not df['Close'].isna().all():
        # Calculate price change over the period
        first_price = df['Close'].dropna().iloc[0]
        last_price = df['Close'].dropna().iloc[-1]
        price_change_pct = ((last_price - first_price) / first_price) * 100
        
        # Simple "Buy the Dip" logic
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

    # 5. Sentiment reasoning
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

    # 6. Final Weighted Score
    # 60% Sentiment, 40% Value
    final_score = (sentiment_score * 0.6) + (value_score * 0.4)
    
    # 7. Create detailed explanation
    explanation = f"{sentiment_reasoning} {price_reasoning}"
    
    return {
        'Sentiment_Score': round(sentiment_score, 1),
        'Value_Score': round(value_score, 1),
        'Final_Score': round(final_score, 1),
        'Explanation': explanation
    }