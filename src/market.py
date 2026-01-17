import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_financials(ticker, sentiment_df):
    """
    Combines the Sentiment Data with Price Data.
    """
    if sentiment_df.empty:
        return sentiment_df
    
    try:
        # Get stock price data for the same date range as sentiment data
        start_date = sentiment_df.index.min()
        end_date = sentiment_df.index.max() + timedelta(days=1)  # Add 1 day to include end date
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        price_data = stock.history(start=start_date, end=end_date)
        
        if price_data.empty:
            print("No price data found for the given date range")
            return sentiment_df
        
        # Convert price data index to date (remove time component)
        price_data.index = price_data.index.date
        
        # Merge sentiment and price data
        merged_df = sentiment_df.join(price_data[['Close', 'Volume']], how='left')
        
        # Forward fill missing price data (for weekends/holidays)
        merged_df['Close'] = merged_df['Close'].fillna(method='ffill')
        merged_df['Volume'] = merged_df['Volume'].fillna(0)
        
        print(f"Successfully merged sentiment data with price data for {len(merged_df)} days")
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