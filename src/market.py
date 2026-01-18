import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import pytz
from functools import lru_cache

# Cache for stock data to avoid repeated API calls
_stock_cache = {}

@lru_cache(maxsize=100)
def get_stock_data(ticker, period="1y"):
    """Fetch stock data using yfinance with caching"""
    cache_key = f"{ticker}_{period}"
    
    # Check if we have recent cached data (within 1 hour)
    if cache_key in _stock_cache:
        cached_data, cache_time = _stock_cache[cache_key]
        if datetime.now() - cache_time < timedelta(hours=1):
            print(f"Using cached stock data for {ticker} ({len(cached_data)} days)")
            return cached_data
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            print(f"No stock data found for {ticker}")
            return pd.DataFrame()
        
        # Cache the data
        _stock_cache[cache_key] = (data, datetime.now())
        
        print(f"Fetched {len(data)} days of stock data for {ticker}")
        return data
    
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=500)
def calculate_z_score_cached(prices_tuple, window=50):
    """Calculate Z-Score with caching - prices as tuple for hashing"""
    prices = pd.Series(prices_tuple)
    return calculate_z_score(prices, window)

def calculate_z_score(prices, window=50):
    """Calculate Z-Score using vectorized numpy operations"""
    if len(prices) < window:
        return 0.0
    
    # Convert to numpy array for faster computation
    prices_array = prices.values
    
    # Vectorized rolling calculations using numpy
    rolling_mean = pd.Series(prices_array).rolling(window=window).mean()
    rolling_std = pd.Series(prices_array).rolling(window=window).std()
    
    # Get the most recent values
    current_price = prices_array[-1]
    current_mean = rolling_mean.iloc[-1]
    current_std = rolling_std.iloc[-1]
    
    if current_std == 0 or np.isnan(current_std):
        return 0.0
    
    z_score = (current_price - current_mean) / current_std
    return float(z_score)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD using vectorized pandas operations"""
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0, False
    
    # Vectorized EMA calculations
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    # MACD Line = Fast EMA - Slow EMA
    macd_line = ema_fast - ema_slow
    
    # Signal Line = EMA of MACD Line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Histogram = MACD - Signal
    histogram = macd_line - signal_line
    
    # Vectorized crossover detection
    bullish_crossover = False
    if len(macd_line) >= 2:
        # Check if MACD crossed above signal line
        prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
        curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
        bullish_crossover = prev_diff <= 0 and curr_diff > 0
    
    return (float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), 
            float(histogram.iloc[-1]), bullish_crossover)

def calculate_rsi(prices, window=14):
    """Calculate RSI using vectorized operations"""
    if len(prices) < window + 1:
        return 50.0  # Neutral RSI
    
    # Vectorized price change calculation
    delta = prices.diff()
    
    # Separate gains and losses using numpy where
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Convert back to Series for rolling operations
    gains_series = pd.Series(gains, index=prices.index)
    losses_series = pd.Series(losses, index=prices.index)
    
    # Vectorized rolling averages
    avg_gains = gains_series.rolling(window=window, min_periods=window).mean()
    avg_losses = losses_series.rolling(window=window, min_periods=window).mean()
    
    # Avoid division by zero
    avg_losses = avg_losses.replace(0, np.finfo(float).eps)
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

@lru_cache(maxsize=200)
def calculate_value_score(z_score):
    """Calculate VALUE strategy score with caching"""
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

@lru_cache(maxsize=200)
def calculate_momentum_score_cached(macd_line, signal_line, histogram, bullish_crossover, rsi):
    """Calculate MOMENTUM strategy score with caching"""
    return calculate_momentum_score(macd_line, signal_line, histogram, bullish_crossover, rsi)

def calculate_momentum_score(macd_line, signal_line, histogram, bullish_crossover, rsi):
    """Calculate MOMENTUM strategy score based on MACD and RSI"""
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

def calculate_verdict(ticker, sentiment_df, strategy="value", lookback_days=30, custom_date=None):
    """Calculate trading verdict with optimized performance"""
    if sentiment_df.empty:
        return {
            'Sentiment_Health': 5.0,
            'Technical_Score': 5.0,
            'Final_Buy_Score': 5.0,
            'Explanation': "No sentiment data available for analysis."
        }
    
    # Optimized period selection using dictionary lookup
    period_map = {
        1: "1d", 2: "2d", 5: "5d", 7: "7d", 
        30: "1mo", 90: "3mo", 180: "6mo"
    }
    
    period = period_map.get(lookback_days, "1y" if lookback_days > 180 else "1mo")
    
    print(f"Fetching {period} of price data for technical analysis...")
    
    # Get stock data with caching
    stock_data = get_stock_data(ticker, period=period)
    if stock_data.empty:
        return {
            'Sentiment_Health': 5.0,
            'Technical_Score': 5.0,
            'Final_Buy_Score': 5.0,
            'Explanation': "No stock price data available for technical analysis."
        }
    
    # Optimized date filtering
    stock_data_filtered = _filter_stock_data(stock_data, lookback_days, custom_date)
    
    if stock_data_filtered.empty:
        print(f"Warning: No price data found within {lookback_days} day timeframe, using available data")
        stock_data_filtered = stock_data
    
    print(f"Using {len(stock_data_filtered)} days of price data (requested: {lookback_days} days)")
    if not stock_data_filtered.empty:
        start_date = stock_data_filtered.index.min()
        end_date = stock_data_filtered.index.max()
        print(f"Price data range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Vectorized sentiment calculation
    avg_sentiment = sentiment_df['Compound_Score'].mean()
    sentiment_health = (avg_sentiment + 1) * 5  # Convert to 0-10 scale
    
    # Get closing prices
    prices = stock_data_filtered['Close']
    
    # Calculate technical indicators based on strategy
    if strategy.lower() == "momentum":
        # MOMENTUM Strategy with caching
        macd_line, signal_line, histogram, bullish_crossover = calculate_macd(prices)
        rsi = calculate_rsi(prices)
        
        technical_score, technical_explanation = calculate_momentum_score_cached(
            macd_line, signal_line, histogram, bullish_crossover, rsi
        )
        
        strategy_name = "MOMENTUM"
        
    else:
        # VALUE Strategy with caching
        z_score = calculate_z_score(prices, window=50)
        technical_score, technical_explanation = calculate_value_score(z_score)
        
        strategy_name = "VALUE"
    
    # Optimized sentiment penalty calculation
    sentiment_penalty = 0.5 if avg_sentiment < -0.5 else 1.0
    penalty_explanation = " FALLING KNIFE WARNING: Negative sentiment reduces score by 50%." if sentiment_penalty < 1.0 else ""
    
    # Vectorized final score calculation
    final_score = ((technical_score * 0.6) + (sentiment_health * 0.4)) * sentiment_penalty
    final_score = np.clip(final_score, 1.0, 10.0)
    
    # Optimized sentiment description using numpy
    sentiment_thresholds = np.array([-0.3, -0.1, 0.1, 0.3])
    sentiment_labels = ["VERY NEGATIVE", "MILDLY NEGATIVE", "NEUTRAL", "MILDLY POSITIVE", "VERY POSITIVE"]
    sentiment_idx = np.searchsorted(sentiment_thresholds, avg_sentiment)
    sentiment_description = sentiment_labels[sentiment_idx]
    
    explanation = f"{strategy_name} Strategy: {technical_explanation} Sentiment is {sentiment_description}.{penalty_explanation}"
    
    return {
        'Sentiment_Health': round(sentiment_health, 1),
        'Technical_Score': round(technical_score, 1),
        'Final_Buy_Score': round(final_score, 1),
        'Explanation': explanation,
        'Strategy': strategy_name
    }

def _filter_stock_data(stock_data, lookback_days, custom_date):
    """Optimized stock data filtering helper function"""
    if lookback_days <= 1:
        # For 1-day analysis, filter to only that specific day
        target_date = custom_date.date() if custom_date else datetime.now().date()
        
        if not stock_data.empty:
            if stock_data.index.tz is not None:
                # Timezone-aware filtering
                target_start = datetime.combine(target_date, datetime.min.time())
                target_end = datetime.combine(target_date, datetime.max.time())
                target_start = pytz.utc.localize(target_start).astimezone(stock_data.index.tz)
                target_end = pytz.utc.localize(target_end).astimezone(stock_data.index.tz)
                return stock_data[(stock_data.index >= target_start) & (stock_data.index <= target_end)]
            else:
                # Non-timezone aware filtering
                return stock_data[stock_data.index.date == target_date]
    else:
        # Multi-day analysis
        cutoff_date = (custom_date or datetime.now()) - timedelta(days=lookback_days)
        
        # Handle timezone awareness
        if not stock_data.empty and stock_data.index.tz is not None:
            if cutoff_date.tzinfo is None:
                cutoff_date = pytz.utc.localize(cutoff_date).astimezone(stock_data.index.tz)
            else:
                cutoff_date = cutoff_date.astimezone(stock_data.index.tz)
        
        return stock_data[stock_data.index >= cutoff_date]
    
    return stock_data

# Legacy function for backward compatibility
def get_financials(ticker, sentiment_df, timeframe_days=30):
    """Legacy function - now just returns sentiment_df for compatibility"""
    return sentiment_df

def get_visualization_data(ticker, sentiment_df, timeframe_days=30):
    """Get combined sentiment and price data for visualization"""
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
        if timeframe_days <= 1:
            # For 1-day analysis, filter to only the specific day from sentiment data
            if period:
                # Get the target date from sentiment data
                target_date = start_date
                if hasattr(target_date, 'date'):
                    target_date = target_date.date()
                
                # Filter price data to only the target date
                if price_data.index.tz is not None:
                    # Convert target date to timezone-aware datetime for comparison
                    target_start = datetime.combine(target_date, datetime.min.time())
                    target_end = datetime.combine(target_date, datetime.max.time())
                    target_start = pd.Timestamp(target_start).tz_localize('UTC').tz_convert(price_data.index.tz)
                    target_end = pd.Timestamp(target_end).tz_localize('UTC').tz_convert(price_data.index.tz)
                    price_data = price_data[(price_data.index >= target_start) & (price_data.index <= target_end)]
                else:
                    # For non-timezone aware data, filter by date
                    price_data = price_data[price_data.index.date == target_date]
        
        # Resample price data to match sentiment frequency
        if timeframe_days <= 1 and interval == "1h":
            price_resampled = price_data.resample('h').last()
        else:
            price_resampled = price_data.resample('D').last()
            price_resampled.index = price_resampled.index.date
        
        start_time = price_data.index.min()
        end_time = price_data.index.max()
        print(f"Original price data: {len(price_data)} records from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Resampled price data: {len(price_resampled)} records")
        
        sentiment_start = sentiment_df.index.min()
        sentiment_end = sentiment_df.index.max()
        if hasattr(sentiment_start, 'strftime'):
            print(f"Sentiment data: {len(sentiment_df)} records from {sentiment_start.strftime('%Y-%m-%d %H:%M')} to {sentiment_end.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"Sentiment data: {len(sentiment_df)} records from {sentiment_start} to {sentiment_end}")
        
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