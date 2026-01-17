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

def calculate_price_derivative(price_series, window=3):
    """Calculate the derivative (slope) of price movement over a window"""
    if len(price_series) < window:
        return 0.0
    
    # Use the last 'window' prices to calculate slope
    recent_prices = price_series.dropna().tail(window)
    if len(recent_prices) < 2:
        return 0.0
    
    # Calculate slope using linear regression over the window
    x = np.arange(len(recent_prices))
    y = recent_prices.values
    
    # Simple slope calculation: (y2 - y1) / (x2 - x1)
    slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0
    
    # Convert to percentage change per day
    avg_price = np.mean(y)
    derivative_pct = (slope / avg_price) * 100 if avg_price > 0 else 0
    
    return derivative_pct

def calculate_value_score(price_change_pct):
    """Calculate value score based on total price change (buy dips strategy)"""
    if price_change_pct < -15:
        return 9.0, f"Price has dropped {abs(price_change_pct):.2f}%, making it a STRONG BUY opportunity."
    elif price_change_pct < -5:
        return 7.0, f"Price has declined {abs(price_change_pct):.2f}%, creating good VALUE."
    elif price_change_pct > 15:
        return 2.0, f"Price has rallied {price_change_pct:.2f}%, making it VERY EXPENSIVE."
    elif price_change_pct > 5:
        return 3.0, f"Price has rallied {price_change_pct:.2f}%, making it EXPENSIVE."
    elif -5 <= price_change_pct <= 5:
        return 6.0, f"Price has moved {price_change_pct:+.2f}%, showing STABLE pricing."
    else:
        return 5.0, f"Price has changed {price_change_pct:+.2f}%."

def calculate_momentum_score(derivative_pct):
    """Calculate momentum score based on price derivative (trend following)"""
    if derivative_pct > 3:
        return 9.0, f"Strong UPWARD momentum ({derivative_pct:+.2f}%/day) - RIDE THE TREND."
    elif derivative_pct > 1:
        return 7.0, f"Positive momentum ({derivative_pct:+.2f}%/day) - GOOD ENTRY POINT."
    elif derivative_pct > -1:
        return 5.0, f"Sideways momentum ({derivative_pct:+.2f}%/day) - NEUTRAL TREND."
    elif derivative_pct > -3:
        return 3.0, f"Negative momentum ({derivative_pct:+.2f}%/day) - DECLINING TREND."
    else:
        return 1.0, f"Strong DOWNWARD momentum ({derivative_pct:+.2f}%/day) - AVOID."

def calculate_verdict(df, strategy_mode="BALANCED"):
    if df.empty:
        return None

    df = df.sort_index(ascending=True)
    
    n = len(df)
    weights = np.linspace(0.2, 1.0, n)
    
    weighted_sentiment = np.average(df['Compound_Score'], weights=weights)
    
    sentiment_score = (weighted_sentiment + 1) * 5
    
    value_score = 5.0
    momentum_score = 5.0
    price_change_pct = 0.0
    derivative_pct = 0.0
    price_reasoning = "No price data available."
    
    if 'Close' in df.columns and not df['Close'].isna().all():
        prices = df['Close'].dropna()
        first_price = prices.iloc[0]
        last_price = prices.iloc[-1]
        
        # Calculate total price change (for value strategy)
        price_change_pct = ((last_price - first_price) / first_price) * 100
        
        # Calculate price derivative (for momentum strategy)
        derivative_pct = calculate_price_derivative(prices, window=min(3, len(prices)))
        
        # Get scores based on strategy
        if strategy_mode == "VALUE":
            value_score, price_reasoning = calculate_value_score(price_change_pct)
            final_score = (sentiment_score * 0.6) + (value_score * 0.4)
            
        elif strategy_mode == "MOMENTUM":
            momentum_score, price_reasoning = calculate_momentum_score(derivative_pct)
            final_score = (sentiment_score * 0.6) + (momentum_score * 0.4)
            
        else:  # BALANCED
            value_score, value_reason = calculate_value_score(price_change_pct)
            momentum_score, momentum_reason = calculate_momentum_score(derivative_pct)
            
            # Combine both approaches
            combined_price_score = (value_score * 0.5) + (momentum_score * 0.5)
            final_score = (sentiment_score * 0.6) + (combined_price_score * 0.4)
            
            price_reasoning = f"VALUE: {value_reason} MOMENTUM: {momentum_reason}"

    else:
        # No price data available - use default scoring
        if strategy_mode == "MOMENTUM":
            final_score = sentiment_score * 0.6 + 5.0 * 0.4
        else:
            final_score = sentiment_score * 0.6 + 5.0 * 0.4

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

    final_score = final_score if 'final_score' in locals() else (sentiment_score * 0.6) + (value_score * 0.4)
    
    explanation = f"{sentiment_reasoning} {price_reasoning}"
    
    # Return appropriate score based on strategy
    if strategy_mode == "MOMENTUM":
        display_score = momentum_score
    else:
        display_score = value_score
    
    return {
        'Sentiment_Score': round(sentiment_score, 1),
        'Value_Score': round(display_score, 1),
        'Final_Score': round(final_score, 1),
        'Explanation': explanation
    }