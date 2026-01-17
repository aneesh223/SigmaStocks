import yfinance as yf
import pandas as pd
from datetime import timedelta

def get_financials(ticker, sentiment_df):
    """
    Downloads stock data matching the sentiment dates and calculates the Buy Algo.
    """
    # Define Range based on sentiment data
    data_start = sentiment_df.index.min()
    data_end = sentiment_df.index.max() + timedelta(days=1)
    
    print(f"Downloading prices from {data_start} to {data_end}...")
    
    # Download Data
    stock_data = yf.download(ticker, start=data_start, end=data_end, progress=False)
    
    # --- FIX 1: FLATTEN MULTI-INDEX COLUMNS ---
    # yfinance sometimes returns columns like ('Close', 'NVDA') instead of just 'Close'
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    # --- FIX 2: NORMALIZE TIMEZONES ---
    # Remove timezone info so we can merge with our simple Date index
    if stock_data.index.tz is not None:
        stock_data.index = stock_data.index.tz_localize(None)

    # Calculate Momentum (Derivative)
    stock_data['Pct_Change'] = stock_data['Close'].pct_change()

    # Merge Sentiment and Price data on the Date index
    merged = pd.merge(sentiment_df, stock_data[['Close', 'Pct_Change']], left_index=True, right_index=True, how='inner')
    
    # Calculate Algo Score (Sentiment - Momentum)
    merged['Buy_Score'] = merged['Compound_Score'] - merged['Pct_Change']
    
    return merged

def calculate_verdict(merged_df):
    """
    Generates a 0-10 Buy Score and text explanation based on the LATEST day of data.
    """
    if merged_df.empty:
        return None

    # Get the latest row of data
    latest = merged_df.iloc[-1]
    
    # --- SCORING MATH ---
    # 1. Normalize Sentiment (-1 to 1) -> (0 to 10)
    sent_score = (latest['Compound_Score'] + 1) * 5
    
    # 2. Normalize Price Action (Target: Buying the Dip)
    raw_pct = latest['Pct_Change']
    # We assume a +/- 5% move is the max "normal" volatility
    value_score = 5 - (raw_pct * 100)
    # Clamp score between 0 and 10
    value_score = max(0, min(10, value_score))
    
    # 3. Weighted Average (60% Sentiment, 40% Value)
    final_score = (sent_score * 0.6) + (value_score * 0.4)
    
    # --- EXPLANATION GENERATION ---
    reasons = []
    
    # Explain Sentiment
    if sent_score >= 7:
        reasons.append("News sentiment is VERY POSITIVE.")
    elif sent_score >= 5:
        reasons.append("News sentiment is NEUTRAL/MILDLY POSITIVE.")
    else:
        reasons.append("News sentiment is NEGATIVE.")
        
    # Explain Price Action
    pct_txt = f"{raw_pct*100:.2f}%"
    if raw_pct <= -0.01:
        reasons.append(f"Price has dropped {pct_txt}, creating a DISCOUNT.")
    elif raw_pct >= 0.01:
        reasons.append(f"Price has rallied {pct_txt}, making it EXPENSIVE.")
    else:
        reasons.append(f"Price is flat ({pct_txt}).")
        
    explanation = " ".join(reasons)
    
    return {
        "Date": latest.name.date(),
        "Sentiment_Score": round(sent_score, 2),
        "Value_Score": round(value_score, 2),
        "Final_Score": round(final_score, 1),
        "Explanation": explanation
    }