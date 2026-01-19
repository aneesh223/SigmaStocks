# Memory-optimized imports - only import what we need
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from functools import lru_cache
import re

# Lazy NLTK loading
def _ensure_nltk_data():
    """Lazy load NLTK data only when needed"""
    try:
        import nltk
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        import nltk
        nltk.download('vader_lexicon', quiet=True)

# Pre-compiled regex patterns for better performance
_SOURCE_TAG_PATTERNS = [
    re.compile(r'^\[.*?\]\s*'),      # Remove from start
    re.compile(r'\s*\[.*?\]$'),      # Remove from end  
    re.compile(r'\s*\[.*?\]\s*')     # Remove from middle
]

# Memory-efficient source categorization using numpy arrays
_PRIMARY_SOURCES_ARRAY = np.array(['SEC EDGAR', 'SEC', 'U.S. Securities and Exchange Commission', 
                                  'Company Investor Relations', 'Investor Relations'])
_INSTITUTIONAL_SOURCES_ARRAY = np.array(['Bloomberg', 'Reuters', 'The Wall Street Journal', 'Financial Times', 
                                        'Barron\'s', 'Morningstar', 'Investor\'s Business Daily'])
_DATA_AGGREGATORS_ARRAY = np.array(['Yahoo Finance', 'MarketWatch', 'Seeking Alpha', 'Zacks', 'TipRanks', 'Benzinga'])
_ENTERTAINMENT_SOURCES_ARRAY = np.array(['The Motley Fool', 'CNBC', 'CNN Business', 'Fox Business', 'MSN',
                                        'USA Today', 'Forbes', 'Fortune', 'Business Insider', 'Reddit',
                                        'Twitter', 'StockTwits', 'Robinhood Snacks', 'TheStreet'])

# Convert to frozensets for O(1) lookup performance
PRIMARY_SOURCES = frozenset(_PRIMARY_SOURCES_ARRAY)
INSTITUTIONAL_SOURCES = frozenset(_INSTITUTIONAL_SOURCES_ARRAY)
DATA_AGGREGATORS = frozenset(_DATA_AGGREGATORS_ARRAY)
ENTERTAINMENT_SOURCES = frozenset(_ENTERTAINMENT_SOURCES_ARRAY)

# Memory-optimized source weights using numpy for faster lookups
_SOURCE_WEIGHT_KEYS = np.array(list(PRIMARY_SOURCES) + list(INSTITUTIONAL_SOURCES) + list(DATA_AGGREGATORS) + list(ENTERTAINMENT_SOURCES))
_SOURCE_WEIGHT_VALUES = np.array([2.0] * len(PRIMARY_SOURCES) + 
                                [1.8, 1.8, 1.8, 1.8, 1.7, 1.6, 1.6] +  # Institutional weights
                                [1.2, 1.2, 1.0, 1.0, 1.0, 1.5] +        # Aggregator weights (Benzinga=1.5)
                                [0.6, 0.8, 0.7, 0.7, 0.6, 0.5, 0.7, 0.7, 0.6, 0.4, 0.3, 0.4, 0.3, 0.5])  # Entertainment

SOURCE_WEIGHTS = dict(zip(_SOURCE_WEIGHT_KEYS, _SOURCE_WEIGHT_VALUES))

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

def remove_source_tags(headline):
    """Remove source tags like [benzinga], [reuters] from headlines"""
    if not headline:
        return headline
    
    # Remove patterns like [source] at the beginning or end of headlines
    cleaned = re.sub(r'^\[.*?\]\s*', '', headline)  # Remove from start
@lru_cache(maxsize=1)
def remove_source_tags(headline):
    """Remove source tags like [benzinga], [reuters] from headlines - optimized with pre-compiled regex"""
    if not headline:
        return headline
    
    cleaned = headline
    for pattern in _SOURCE_TAG_PATTERNS:
        cleaned = pattern.sub('', cleaned)
    
    return cleaned.strip()

def get_vader_analyzer():
    """Get cached VADER analyzer with financial lexicon - lazy initialization"""
    global _vader_analyzer
    if _vader_analyzer is None:
        _ensure_nltk_data()  # Lazy load NLTK data
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
    """Optimized source categorization with O(1) lookups and case-insensitive matching"""
    # Convert to title case for consistent matching
    source_title = source.title()
    
    if source_title in PRIMARY_SOURCES:
        return 'primary'
    elif source_title in INSTITUTIONAL_SOURCES:
        return 'institutional'
    elif source_title in DATA_AGGREGATORS:
        return 'aggregator'
    elif source_title in ENTERTAINMENT_SOURCES:
        return 'entertainment'
    else:
        # Fast keyword matching (case-insensitive)
        source_lower = source.lower()
        if any(kw in source_lower for kw in ('sec', 'securities', 'investor relations')):
            return 'primary'
        elif any(kw in source_lower for kw in ('bloomberg', 'reuters', 'wall street', 'financial times')):
            return 'institutional'
        elif any(kw in source_lower for kw in ('yahoo finance', 'marketwatch', 'seeking alpha', 'benzinga')):
            return 'aggregator'
        else:
            return 'entertainment'
def get_sentiment(parsed_data, ticker, timeframe_days=30):
    """Optimized sentiment analysis with vectorized operations and memory efficiency"""
    if not parsed_data:
        return pd.DataFrame()

    # Create DataFrame with optimized dtypes for memory efficiency
    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert timestamps to datetime if needed (vectorized)
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    
    # Get cached VADER analyzer
    vader = get_vader_analyzer()
    
    # Vectorized headline cleaning using numpy operations
    df['Cleaned_Headline'] = df['Headline'].apply(clean_headline)
    
    # Batch sentiment analysis with list comprehension (faster than apply)
    raw_scores = np.array([vader.polarity_scores(headline)['compound'] for headline in df['Cleaned_Headline']])
    df['Raw_Score'] = raw_scores
    
    # Vectorized target validation using numpy where for better performance
    valid_mask = df.apply(lambda row: validate_target(row['Headline'], ticker, row['Raw_Score']), axis=1)
    df['Raw_Score'] = valid_mask
    
    # Optimized source categorization with vectorized operations
    df['Source_Type'] = df['Source'].apply(categorize_source)
    
    # Vectorized weight calculation using numpy for better performance
    source_weights = df['Source'].map(lambda s: SOURCE_WEIGHTS.get(s.title(), SOURCE_WEIGHTS.get(s, 1.0)))
    df['Weight'] = source_weights
    df['Compound_Score'] = np.clip(df['Raw_Score'] * df['Weight'], -1.0, 1.0)
    
    # Optimized time grouping
    if timeframe_days <= 1:
        df['TimeGroup'] = df['Timestamp'].dt.floor('h')
    else:
        df['TimeGroup'] = df['Timestamp'].dt.date
    
    # Pre-compute masks for efficiency (avoid repeated boolean operations)
    source_masks = {
        'primary': df['Source_Type'] == 'primary',
        'institutional': df['Source_Type'] == 'institutional', 
        'aggregator': df['Source_Type'] == 'aggregator',
        'entertainment': df['Source_Type'] == 'entertainment'
    }
    
    # Optimized weighted mean function using numpy
    def weighted_mean_fast(group):
        weights = group['Weight'].values
        scores = group['Compound_Score'].values
        weight_sum = np.sum(weights)
        return np.sum(scores * weights) / weight_sum if weight_sum > 0 else 0
    
    # Main aggregation with optimized groupby operations
    result_df = df.groupby('TimeGroup', sort=False).agg({
        'Raw_Score': 'mean', 
        'Weight': 'mean'
    })
    
    # Calculate weighted compound scores using optimized function
    try:
        weighted_scores = df.groupby('TimeGroup', sort=False).apply(weighted_mean_fast, include_groups=False)
    except TypeError:
        weighted_scores = df.groupby('TimeGroup', sort=False).apply(weighted_mean_fast)
    
    result_df['Compound_Score'] = weighted_scores
    
    # Efficient sentiment and count calculations using pre-computed masks
    for source_type, mask in source_masks.items():
        if mask.any():  # Only process if there are articles of this type
            source_df = df[mask]
            sentiment_col = f'{source_type.title()}_Sentiment'
            count_col = f'{source_type.title()}_Count'
            
            try:
                sentiment_scores = source_df.groupby('TimeGroup', sort=False).apply(weighted_mean_fast, include_groups=False)
            except TypeError:
                sentiment_scores = source_df.groupby('TimeGroup', sort=False).apply(weighted_mean_fast)
            
            counts = source_df.groupby('TimeGroup', sort=False).size()
            
            result_df[sentiment_col] = sentiment_scores.reindex(result_df.index, fill_value=0)
            result_df[count_col] = counts.reindex(result_df.index, fill_value=0)
        else:
            # Set to zero if no articles of this type
            result_df[f'{source_type.title()}_Sentiment'] = 0
            result_df[f'{source_type.title()}_Count'] = 0
    
    # Print simple summary (no source breakdown)
    print(f"Analyzed {len(df)} news articles for sentiment")
    
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
        # Remove source tags like [benzinga] from headlines
        cleaned_headline = remove_source_tags(top_story['Headline'])
        best_headline = f"{cleaned_headline} (Score: {top_story['Score']:.2f})"
    else:
        best_headline = "No significant positive news found."

    if bottom_story['Score'] < -0.1:
        # Remove source tags like [benzinga] from headlines
        cleaned_headline = remove_source_tags(bottom_story['Headline'])
        worst_headline = f"{cleaned_headline} (Score: {bottom_story['Score']:.2f})"
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