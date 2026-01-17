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
    
    # Calculate weighted sentiment for overall, insider, and consumer
    weighted_sentiment = np.average(df['Compound_Score'], weights=weights)
    
def calculate_verdict(df, strategy_mode="BALANCED"):
    if df.empty:
        return None

    df = df.sort_index(ascending=True)
    
    n = len(df)
    weights = np.linspace(0.2, 1.0, n)
    
    # Calculate weighted sentiment for overall
    weighted_sentiment = np.average(df['Compound_Score'], weights=weights)
    
    # Calculate separate sentiment for each source tier with minimum article thresholds
    MIN_ARTICLES_THRESHOLD = 3  # Minimum articles needed to show a category
    
    primary_sentiment = 0.0
    institutional_sentiment = 0.0
    aggregator_sentiment = 0.0
    entertainment_sentiment = 0.0
    
    # Track which categories have sufficient data
    valid_categories = {}
    
    if 'Primary_Sentiment' in df.columns and not df['Primary_Sentiment'].isna().all():
        primary_mask = ~df['Primary_Sentiment'].isna()
        primary_article_count = df['Primary_Count'].sum() if 'Primary_Count' in df.columns else 0
        if primary_mask.any() and primary_article_count >= MIN_ARTICLES_THRESHOLD:
            primary_sentiment = np.average(df.loc[primary_mask, 'Primary_Sentiment'], 
                                         weights=weights[primary_mask])
            valid_categories['Primary'] = (primary_sentiment, primary_article_count)
    
    if 'Institutional_Sentiment' in df.columns and not df['Institutional_Sentiment'].isna().all():
        institutional_mask = ~df['Institutional_Sentiment'].isna()
        institutional_article_count = df['Institutional_Count'].sum() if 'Institutional_Count' in df.columns else 0
        if institutional_mask.any() and institutional_article_count >= MIN_ARTICLES_THRESHOLD:
            institutional_sentiment = np.average(df.loc[institutional_mask, 'Institutional_Sentiment'], 
                                               weights=weights[institutional_mask])
            valid_categories['Institutional'] = (institutional_sentiment, institutional_article_count)
    
    if 'Aggregator_Sentiment' in df.columns and not df['Aggregator_Sentiment'].isna().all():
        aggregator_mask = ~df['Aggregator_Sentiment'].isna()
        aggregator_article_count = df['Aggregator_Count'].sum() if 'Aggregator_Count' in df.columns else 0
        if aggregator_mask.any() and aggregator_article_count >= MIN_ARTICLES_THRESHOLD:
            aggregator_sentiment = np.average(df.loc[aggregator_mask, 'Aggregator_Sentiment'], 
                                            weights=weights[aggregator_mask])
            valid_categories['Aggregator'] = (aggregator_sentiment, aggregator_article_count)
    
    if 'Entertainment_Sentiment' in df.columns and not df['Entertainment_Sentiment'].isna().all():
        entertainment_mask = ~df['Entertainment_Sentiment'].isna()
        entertainment_article_count = df['Entertainment_Count'].sum() if 'Entertainment_Count' in df.columns else 0
        if entertainment_mask.any() and entertainment_article_count >= MIN_ARTICLES_THRESHOLD:
            entertainment_sentiment = np.average(df.loc[entertainment_mask, 'Entertainment_Sentiment'], 
                                               weights=weights[entertainment_mask])
            valid_categories['Entertainment'] = (entertainment_sentiment, entertainment_article_count)
    
    # Convert to 0-10 scale only for valid categories
    sentiment_score = (weighted_sentiment + 1) * 5
    
    # Only calculate scores for categories with sufficient data
    category_scores = {}
    if 'Primary' in valid_categories:
        category_scores['Primary_Score'] = (valid_categories['Primary'][0] + 1) * 5
    if 'Institutional' in valid_categories:
        category_scores['Institutional_Score'] = (valid_categories['Institutional'][0] + 1) * 5
    if 'Aggregator' in valid_categories:
        category_scores['Aggregator_Score'] = (valid_categories['Aggregator'][0] + 1) * 5
    if 'Entertainment' in valid_categories:
        category_scores['Entertainment_Score'] = (valid_categories['Entertainment'][0] + 1) * 5
    
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

    # Add source tier analysis - only for categories with sufficient data
    sentiment_breakdown = ""
    
    if len(valid_categories) >= 2:
        # Sort by sentiment value
        sorted_categories = sorted(valid_categories.items(), key=lambda x: x[1][0], reverse=True)
        highest_name, (highest_sentiment, highest_count) = sorted_categories[0]
        lowest_name, (lowest_sentiment, lowest_count) = sorted_categories[-1]
        
        sentiment_diff = highest_sentiment - lowest_sentiment
        
        if sentiment_diff > 0.3:
            sentiment_breakdown = f" {highest_name.upper()} sources ({highest_count} articles) much more bullish than {lowest_name.upper()} ({lowest_count} articles)."
        elif sentiment_diff > 0.15:
            sentiment_breakdown = f" {highest_name.upper()} sources more optimistic than {lowest_name.upper()}."
        else:
            category_names = [name.upper() for name in valid_categories.keys()]
            sentiment_breakdown = f" {', '.join(category_names)} sources showing ALIGNED sentiment."
    
    elif len(valid_categories) == 1:
        category_name, (_, count) = list(valid_categories.items())[0]
        sentiment_breakdown = f" Coverage primarily from {category_name.upper()} sources ({count} articles)."
    
    else:
        sentiment_breakdown = " Insufficient reliable news coverage for analysis."

    final_score = final_score if 'final_score' in locals() else (sentiment_score * 0.6) + (value_score * 0.4)
    
    explanation = f"{sentiment_reasoning}{sentiment_breakdown} {price_reasoning}"
    
    # Return appropriate score based on strategy
    if strategy_mode == "MOMENTUM":
        display_score = momentum_score
    else:
        display_score = value_score
    
    # Build return dictionary with only valid categories
    result = {
        'Sentiment_Score': round(sentiment_score, 1),
        'Value_Score': round(display_score, 1),
        'Final_Score': round(final_score, 1),
        'Explanation': explanation,
        'Valid_Categories': list(valid_categories.keys())  # Track which categories are valid
    }
    
    # Add scores only for valid categories
    for category, score in category_scores.items():
        result[category] = round(score, 1)
    
    return result