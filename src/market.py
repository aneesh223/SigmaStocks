import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(ticker, period="1y"):
    """
    Fetch stock data using yfinance
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period (1y, 6mo, 3mo, etc.)
    
    Returns:
        pd.DataFrame: Stock price data with OHLCV
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            print(f"No stock data found for {ticker}")
            return pd.DataFrame()
        
        print(f"Fetched {len(data)} days of stock data for {ticker}")
        return data
    
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_z_score(prices, window=50):
    """
    Calculate Z-Score based on moving average and standard deviation
    
    Args:
        prices (pd.Series): Price series
        window (int): Rolling window for calculation
    
    Returns:
        float: Current Z-Score
    """
    if len(prices) < window:
        return 0.0
    
    # Calculate rolling mean and std
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    
    # Calculate Z-Score for the most recent price
    current_price = prices.iloc[-1]
    current_mean = rolling_mean.iloc[-1]
    current_std = rolling_std.iloc[-1]
    
    if current_std == 0:
        return 0.0
    
    z_score = (current_price - current_mean) / current_std
    return z_score

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices (pd.Series): Price series
        fast (int): Fast EMA period
        slow (int): Slow EMA period  
        signal (int): Signal line EMA period
    
    Returns:
        tuple: (macd_line, signal_line, histogram, bullish_crossover)
    """
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0, False
    
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    # MACD Line = Fast EMA - Slow EMA
    macd_line = ema_fast - ema_slow
    
    # Signal Line = 9-day EMA of MACD Line
    signal_line = macd_line.ewm(span=signal).mean()
    
    # Histogram = MACD - Signal
    histogram = macd_line - signal_line
    
    # Check for bullish crossover (MACD crosses above Signal)
    bullish_crossover = False
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        curr_macd = macd_line.iloc[-1]
        curr_signal = signal_line.iloc[-1]
        
        # Bullish crossover: MACD was below signal, now above
        bullish_crossover = (prev_macd <= prev_signal) and (curr_macd > curr_signal)
    
    return curr_macd, curr_signal, histogram.iloc[-1], bullish_crossover

def calculate_rsi(prices, window=14):
    """
    Calculate RSI (Relative Strength Index)
    
    Args:
        prices (pd.Series): Price series
        window (int): RSI calculation window
    
    Returns:
        float: Current RSI value
    """
    if len(prices) < window + 1:
        return 50.0  # Neutral RSI
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1]

def calculate_value_score(z_score):
    """
    Calculate VALUE strategy score based on Z-Score
    
    Args:
        z_score (float): Current Z-Score
    
    Returns:
        tuple: (technical_score, explanation)
    """
    # VALUE Strategy: Look for oversold conditions (Z < -2.0)
    if z_score <= -2.5:
        technical_score = 9.0
        explanation = f"Stock is {abs(z_score):.1f} standard deviations below mean - DEEP VALUE opportunity."
    elif z_score <= -2.0:
        technical_score = 8.0
        explanation = f"Stock is {abs(z_score):.1f} standard deviations below mean - Strong BUY signal."
    elif z_score <= -1.5:
        technical_score = 7.0
        explanation = f"Stock is {abs(z_score):.1f} standard deviations below mean - Good VALUE entry."
    elif z_score <= -1.0:
        technical_score = 6.0
        explanation = f"Stock is {abs(z_score):.1f} standard deviations below mean - Moderate VALUE."
    elif z_score <= 0.5:
        technical_score = 5.0
        explanation = f"Stock is near mean (Z-Score: {z_score:.1f}) - NEUTRAL valuation."
    elif z_score <= 1.5:
        technical_score = 4.0
        explanation = f"Stock is {z_score:.1f} standard deviations above mean - Slightly EXPENSIVE."
    elif z_score <= 2.0:
        technical_score = 3.0
        explanation = f"Stock is {z_score:.1f} standard deviations above mean - EXPENSIVE."
    else:
        technical_score = 2.0
        explanation = f"Stock is {z_score:.1f} standard deviations above mean - VERY EXPENSIVE."
    
    return technical_score, explanation

def calculate_momentum_score(macd_line, signal_line, histogram, bullish_crossover, rsi):
    """
    Calculate MOMENTUM strategy score based on MACD and RSI
    
    Args:
        macd_line (float): Current MACD value
        signal_line (float): Current Signal line value
        histogram (float): MACD histogram value
        bullish_crossover (bool): Whether bullish crossover occurred
        rsi (float): Current RSI value
    
    Returns:
        tuple: (technical_score, explanation)
    """
    # Base score starts at 5 (neutral)
    technical_score = 5.0
    explanations = []
    
    # MACD Analysis
    if bullish_crossover:
        if rsi < 70:  # Not overbought
            technical_score += 3.0
            explanations.append("MACD bullish crossover detected")
        else:
            technical_score += 1.0
            explanations.append("MACD bullish crossover but RSI overbought")
    elif macd_line > signal_line:
        if histogram > 0:
            technical_score += 2.0
            explanations.append("MACD above signal with positive momentum")
        else:
            technical_score += 1.0
            explanations.append("MACD above signal but weakening")
    else:
        if histogram < 0:
            technical_score -= 1.0
            explanations.append("MACD below signal with negative momentum")
    
    # RSI Analysis
    if rsi < 30:
        technical_score += 1.0
        explanations.append(f"RSI oversold at {rsi:.0f}")
    elif rsi > 70:
        technical_score -= 1.0
        explanations.append(f"RSI overbought at {rsi:.0f}")
    else:
        explanations.append(f"RSI neutral at {rsi:.0f}")
    
    # Cap the score between 1-10
    technical_score = max(1.0, min(10.0, technical_score))
    
    explanation = ". ".join(explanations) + "."
    
    return technical_score, explanation

def calculate_verdict(ticker, sentiment_df, strategy="value"):
    """
    Calculate trading verdict based on technical analysis and sentiment
    
    Args:
        ticker (str): Stock ticker symbol
        sentiment_df (pd.DataFrame): Sentiment data from analyzer
        strategy (str): Trading strategy - 'value' or 'momentum'
    
    Returns:
        dict: Trading verdict with scores and explanation
    """
    if sentiment_df.empty:
        return {
            'Sentiment_Health': 5.0,
            'Technical_Score': 5.0,
            'Final_Buy_Score': 5.0,
            'Explanation': "No sentiment data available for analysis."
        }
    
    # Get stock data
    stock_data = get_stock_data(ticker, period="1y")
    if stock_data.empty:
        return {
            'Sentiment_Health': 5.0,
            'Technical_Score': 5.0,
            'Final_Buy_Score': 5.0,
            'Explanation': "No stock price data available for technical analysis."
        }
    
    # Calculate sentiment health (convert from -1 to 1 scale to 0 to 10 scale)
    avg_sentiment = sentiment_df['Compound_Score'].mean()
    sentiment_health = (avg_sentiment + 1) * 5  # Convert to 0-10 scale
    
    # Get closing prices
    prices = stock_data['Close']
    
    # Calculate technical indicators based on strategy
    if strategy.lower() == "momentum":
        # MOMENTUM Strategy
        macd_line, signal_line, histogram, bullish_crossover = calculate_macd(prices)
        rsi = calculate_rsi(prices)
        
        technical_score, technical_explanation = calculate_momentum_score(
            macd_line, signal_line, histogram, bullish_crossover, rsi
        )
        
        strategy_name = "MOMENTUM"
        
    else:
        # VALUE Strategy (default)
        z_score = calculate_z_score(prices, window=50)
        technical_score, technical_explanation = calculate_value_score(z_score)
        
        strategy_name = "VALUE"
    
    # Apply sentiment penalty for negative sentiment (falling knife protection)
    sentiment_penalty = 1.0
    if avg_sentiment < -0.5:
        sentiment_penalty = 0.5  # 50% penalty for very negative sentiment
        penalty_explanation = " FALLING KNIFE WARNING: Negative sentiment reduces score by 50%."
    else:
        penalty_explanation = ""
    
    # Calculate final score (60% technical, 40% sentiment, with penalty)
    final_score = ((technical_score * 0.6) + (sentiment_health * 0.4)) * sentiment_penalty
    final_score = max(1.0, min(10.0, final_score))  # Cap between 1-10
    
    # Build explanation
    if avg_sentiment > 0.3:
        sentiment_description = "VERY POSITIVE"
    elif avg_sentiment > 0.1:
        sentiment_description = "MILDLY POSITIVE"
    elif avg_sentiment > -0.1:
        sentiment_description = "NEUTRAL"
    elif avg_sentiment > -0.3:
        sentiment_description = "MILDLY NEGATIVE"
    else:
        sentiment_description = "VERY NEGATIVE"
    
    explanation = f"{strategy_name} Strategy: {technical_explanation} Sentiment is {sentiment_description}.{penalty_explanation}"
    
    return {
        'Sentiment_Health': round(sentiment_health, 1),
        'Technical_Score': round(technical_score, 1),
        'Final_Buy_Score': round(final_score, 1),
        'Explanation': explanation,
        'Strategy': strategy_name
    }

# Legacy function for backward compatibility
def get_financials(ticker, sentiment_df, timeframe_days=30):
    """
    Legacy function - now just returns sentiment_df for compatibility
    """
    return sentiment_df

def get_visualization_data(ticker, sentiment_df, timeframe_days=30):
    """
    Get combined sentiment and price data for visualization
    
    Args:
        ticker (str): Stock ticker symbol
        sentiment_df (pd.DataFrame): Sentiment data from analyzer
        timeframe_days (int): Number of days to look back
    
    Returns:
        pd.DataFrame: Combined sentiment and price data for plotting
    """
    if sentiment_df.empty:
        return sentiment_df
    
    try:
        # Get the date range from sentiment data
        start_date = sentiment_df.index.min()
        end_date = sentiment_df.index.max()
        
        # Handle different index types
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(end_date, 'date'):
            end_date = end_date.date()
            
        # Add buffer for price data
        end_date = end_date + timedelta(days=1)
        
        # Determine interval based on timeframe
        if timeframe_days <= 1:
            interval = "1h"
            period = "2d"
        else:
            interval = "1d"
            period = None
        
        # Get stock data
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
        
        # Filter price data for intraday if needed
        if timeframe_days <= 1 and period:
            start_timestamp = pd.Timestamp(start_date).tz_localize('UTC')
            if price_data.index.tz is not None:
                start_timestamp = start_timestamp.tz_convert(price_data.index.tz)
            price_data = price_data[price_data.index >= start_timestamp]
        
        # Resample price data to match sentiment frequency
        if timeframe_days <= 1 and interval == "1h":
            price_resampled = price_data.resample('h').last()
        else:
            price_resampled = price_data.resample('D').last()
            price_resampled.index = price_resampled.index.date
        
        print(f"Original price data: {len(price_data)} records from {price_data.index.min()} to {price_data.index.max()}")
        print(f"Resampled price data: {len(price_resampled)} records")
        print(f"Sentiment data: {len(sentiment_df)} records from {sentiment_df.index.min()} to {sentiment_df.index.max()}")
        
        # Merge sentiment and price data
        merged_df = sentiment_df.join(price_resampled[['Close', 'Volume']], how='left')
        
        # Forward fill missing prices
        merged_df['Close'] = merged_df['Close'].ffill()
        merged_df['Volume'] = merged_df['Volume'].fillna(0)
        
        print(f"Successfully merged sentiment data with price data for {len(merged_df)} time periods")
        return merged_df
        
    except Exception as e:
        print(f"Error fetching price data for visualization: {e}")
        return sentiment_df