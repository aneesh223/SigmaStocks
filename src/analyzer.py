import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
import numpy as np
from functools import lru_cache

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Convert to frozensets for O(1) lookup performance
PRIMARY_SOURCES = frozenset({
    'SEC EDGAR', 'SEC', 'U.S. Securities and Exchange Commission', 
    'Company Investor Relations', 'Investor Relations'
})

INSTITUTIONAL_SOURCES = frozenset({
    'Bloomberg', 'Reuters', 'The Wall Street Journal', 'Financial Times', 
    'Barron\'s', 'Morningstar', 'Investor\'s Business Daily'
})

DATA_AGGREGATORS = frozenset({
    'Yahoo Finance', 'MarketWatch', 'Seeking Alpha', 'Zacks', 'TipRanks'
})

ENTERTAINMENT_SOURCES = frozenset({
    'The Motley Fool', 'CNBC', 'CNN Business', 'Fox Business', 'MSN',
    'USA Today', 'Forbes', 'Fortune', 'Business Insider', 'Reddit',
    'Twitter', 'StockTwits', 'Robinhood Snacks', 'Benzinga', 'TheStreet'
})

# Optimized source weights with dictionary comprehension
SOURCE_WEIGHTS = {
    **{source: 2.0 for source in PRIMARY_SOURCES},
    'Bloomberg': 1.8, 'Reuters': 1.8, 'The Wall Street Journal': 1.8, 
    'Financial Times': 1.8, 'Barron\'s': 1.7, 'Morningstar': 1.6,
    'Investor\'s Business Daily': 1.6,
    'Yahoo Finance': 1.2, 'MarketWatch': 1.2, 'Seeking Alpha': 1.0, 
    'Zacks': 1.0, 'TipRanks': 1.0,
    'The Motley Fool': 0.6, 'CNBC': 0.8, 'CNN Business': 0.7, 
    'Fox Business': 0.7, 'MSN': 0.6, 'USA Today': 0.5, 'Forbes': 0.7,
    'Fortune': 0.7, 'Business Insider': 0.6, 'Reddit': 0.4, 'Twitter': 0.3,
    'StockTwits': 0.4, 'Robinhood Snacks': 0.3, 'Benzinga': 0.5, 'TheStreet': 0.5
}

# Enhanced financial lexicon for better sentiment accuracy
FINANCIAL_LEXICON = {
    'error': 0.0, 'loom': 0.0, 'vice': 0.0, 'gross': 0.0, 'mine': 0.0, 
    'arrest': 0.0, 'fool': 0.0, 'motley': 0.0,
    'tank': -2.5, 'plunge': -3.0, 'brutal': -3.0, 'crash': -3.0, 'collapse': -3.0,
    'miss': -2.5, 'fall short': -2.5, 'falls short': -2.5, 'fell short': -2.5,
    'disappointing': -2.0, 'decline': -2.0, 'drop': -2.0, 'tumble': -2.5,
    'slump': -2.5, 'weak': -1.5, 'poor': -2.0, 'loss': -2.0, 'deficit': -2.0,
    'flat': -1.5, 'stagnant': -1.5, 'choppy': -1.0, 'sideways': -1.0,
    'beat': 2.5, 'crush': 3.0, 'soar': 3.0, 'skyrocket': 3.5, 'rally': 2.5,
    'exceed': 2.0, 'outperform': 2.0, 'strong': 2.0, 'robust': 2.0,
    'growth': 1.5, 'gains': 2.0, 'profit': 2.0, 'upgrade': 2.0, 'bullish': 2.5,
    'surprise': 1.5, 'revenue': 1.0, 'earnings': 1.0, 'call': 1.0, 'shorting': -2.0
}

# Cache VADER analyzer instance
_vader_analyzer = None

def get_vader_analyzer():
    """Get cached VADER analyzer with financial lexicon"""
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = sia()
        _vader_analyzer.lexicon.update(FINANCIAL_LEXICON)
    return _vader_analyzer

@lru_cache(maxsize=1000)
def clean_headline(text):
    """Optimized headline cleaning with caching"""
    text_lower = text.lower()
    
    # Optimized replacements with early returns
    replacements = {
        "earnings fall short": "disappointing earnings",
        "falls short": "disappointing", "fell short": "disappointing",
        "revenue tops estimates": "strong revenue", "tops estimates": "beat expectations",
        "exceeds expectations": "beat expectations", "below estimates": "miss expectations",
        "break losing streak": "rally", "snaps losing streak": "rally",
        "recovers from": "rally", "bounce back": "rally",
        "top pick": "best stock", "buy rating": "perfect", "strong buy": "perfect",
        "outperform": "winner", "stock to buy": "good stock"
    }
    
    for phrase, replacement in replacements.items():
        if phrase in text_lower:
            text_lower = text_lower.replace(phrase, replacement)
    
    return text_lower

@lru_cache(maxsize=1000)
def validate_target(headline, ticker, raw_score):
    """Optimized target validation with caching"""
    headline_lower = headline.lower()
    ticker_lower = ticker.lower()
    
    if " for " in headline_lower:
        parts = headline_lower.split(" for ", 1)
        if len(parts) > 1:
            target = parts[1].split()[0].strip(".,;:!?")
            
            # Use frozenset for O(1) lookup
            generics = frozenset(['investors', 'shareholders', 'holders', 'stock', 'market', 
                                'trading', 'tech', 'semis', 'sector', 'growth', 'economy'])
            
            if target != ticker_lower and target not in generics:
                return 0.0
    
    return raw_score

@lru_cache(maxsize=500)
def categorize_source(source):
    """Optimized source categorization with O(1) lookups"""
    if source in PRIMARY_SOURCES:
        return 'primary'
    elif source in INSTITUTIONAL_SOURCES:
        return 'institutional'
    elif source in DATA_AGGREGATORS:
        return 'aggregator'
    elif source in ENTERTAINMENT_SOURCES:
        return 'entertainment'
    else:
        # Fast keyword matching
        source_lower = source.lower()
        if any(kw in source_lower for kw in ('sec', 'securities', 'investor relations')):
            return 'primary'
        elif any(kw in source_lower for kw in ('bloomberg', 'reuters', 'wall street', 'financial times')):
            return 'institutional'
        elif any(kw in source_lower for kw in ('yahoo finance', 'marketwatch', 'seeking alpha')):
            return 'aggregator'
        else:
            return 'entertainment'
def get_sentiment(parsed_data, ticker, timeframe_days=30):
    """Optimized sentiment analysis with vectorized operations"""
    if not parsed_data:
        return pd.DataFrame()

    # Create DataFrame with optimized dtypes
    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    if df.empty:
        return pd.DataFrame()
    
    # Get cached VADER analyzer
    vader = get_vader_analyzer()
    
    # Vectorized headline cleaning
    df['Cleaned_Headline'] = df['Headline'].apply(clean_headline)
    
    # Batch sentiment analysis
    raw_scores = [vader.polarity_scores(headline)['compound'] for headline in df['Cleaned_Headline']]
    df['Raw_Score'] = raw_scores
    
    # Vectorized target validation
    df['Raw_Score'] = df.apply(lambda row: validate_target(row['Headline'], ticker, row['Raw_Score']), axis=1)
    
    # Optimized source categorization
    df['Source_Type'] = df['Source'].apply(categorize_source)
    
    # Vectorized weight calculation using map for better performance
    df['Weight'] = df['Source'].map(SOURCE_WEIGHTS).fillna(1.0)
    df['Compound_Score'] = (df['Raw_Score'] * df['Weight']).clip(-1.0, 1.0)
    
    # Optimized time grouping
    if timeframe_days <= 1:
        df['TimeGroup'] = df['Timestamp'].dt.floor('h')
    else:
        df['TimeGroup'] = df['Timestamp'].dt.date
    
    # Pre-filter DataFrames for efficiency
    source_masks = {
        'primary': df['Source_Type'] == 'primary',
        'institutional': df['Source_Type'] == 'institutional', 
        'aggregator': df['Source_Type'] == 'aggregator',
        'entertainment': df['Source_Type'] == 'entertainment'
    }
    
    # Main aggregation
    result_df = df.groupby('TimeGroup').agg({
        'Compound_Score': 'mean',
        'Raw_Score': 'mean', 
        'Weight': 'mean'
    })
    
    # Efficient sentiment and count calculations
    for source_type, mask in source_masks.items():
        source_df = df[mask]
        sentiment_col = f'{source_type.title()}_Sentiment'
        count_col = f'{source_type.title()}_Count'
        
        if not source_df.empty:
            sentiment_scores = source_df.groupby('TimeGroup')['Compound_Score'].mean()
            counts = source_df.groupby('TimeGroup').size()
            
            result_df[sentiment_col] = sentiment_scores.reindex(result_df.index, fill_value=0)
            result_df[count_col] = counts.reindex(result_df.index, fill_value=0)
        else:
            result_df[sentiment_col] = 0
            result_df[count_col] = 0
    
    # Optimized summary print
    source_counts = df['Source_Type'].value_counts()
    print(f"Source breakdown - " + 
          ", ".join([f"{k.title()}: {source_counts.get(k, 0)}" for k in ['primary', 'institutional', 'aggregator', 'entertainment']]))
    
    return result_df

@lru_cache(maxsize=1000)
def get_top_headlines(parsed_data_tuple, ticker, timeframe_days=30):
    """Optimized top headlines extraction with caching"""
    if not parsed_data_tuple:
        return "N/A", "N/A"

    # Convert tuple back to list for DataFrame creation
    parsed_data = list(parsed_data_tuple)
    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    if df.empty:
        return "No significant news.", "No significant news."
    
    # Get cached VADER analyzer
    vader = get_vader_analyzer()
    
    # Vectorized operations
    df['Cleaned'] = df['Headline'].apply(clean_headline)
    df['Score'] = df['Cleaned'].apply(lambda title: vader.polarity_scores(title)['compound'])
    df['Score'] = df.apply(lambda row: validate_target(row['Headline'], ticker, row['Score']), axis=1)

    # Sort once for efficiency
    df = df.sort_values(by='Score', ascending=False)
    
    top_story = df.iloc[0]
    bottom_story = df.iloc[-1]
    
    if top_story['Score'] > 0.1:
        best_headline = f"[{top_story['Source']}] {top_story['Headline']} (Score: {top_story['Score']:.2f})"
    else:
        best_headline = "No significant positive news found."

    if bottom_story['Score'] < -0.1:
        worst_headline = f"[{bottom_story['Source']}] {bottom_story['Headline']} (Score: {bottom_story['Score']:.2f})"
    else:
        worst_headline = "No significant negative news found."
    
    return best_headline, worst_headline

# Wrapper function to handle the tuple conversion for caching
def get_top_headlines_wrapper(parsed_data, ticker, timeframe_days=30):
    """Wrapper to convert list to tuple for caching"""
    if not parsed_data:
        return "N/A", "N/A"
    
    # Convert to tuple for hashing (required for caching)
    parsed_data_tuple = tuple(tuple(row) if isinstance(row, list) else row for row in parsed_data)
    return get_top_headlines(parsed_data_tuple, ticker, timeframe_days)