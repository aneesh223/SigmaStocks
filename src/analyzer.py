# Memory-optimized imports - only import what we need
import pandas as pd
import numpy as np
from functools import lru_cache
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# VADER sentiment analysis imports
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not available, using fallback sentiment analysis")

# DistilRoBERTa model imports - lazy loaded for better performance
_ai_model = None
_ai_tokenizer = None

def _ensure_model():
    """Lazy load DistilRoBERTa model and tokenizer only when needed"""
    global _ai_model, _ai_tokenizer
    if _ai_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            print("Loading DistilRoBERTa financial sentiment model (first time only)...")
            model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
            _ai_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _ai_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Check for GPU availability and move model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                _ai_model = _ai_model.to(device)
                print(f"✅ DistilRoBERTa loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("✅ DistilRoBERTa loaded on CPU")
            
            # Set to evaluation mode for inference
            _ai_model.eval()
            
        except Exception as e:
            print(f"Error loading DistilRoBERTa model: {e}")
            print("Falling back to VADER-only analysis...")
            return False
    return True

# Pre-compiled regex patterns for better performance
_SOURCE_TAG_PATTERNS = [
    re.compile(r'^\[.*?\]\s*'),      # Remove from start
    re.compile(r'\s*\[.*?\]$'),      # Remove from end  
    re.compile(r'\s*\[.*?\]\s*')     # Remove from middle (with proper spacing)
]

# Enhanced Financial Lexicon for VADER
FINANCIAL_LEXICON = {
    'error': 0.0, 'loom': 0.0, 'vice': 0.0, 'gross': 0.0, 'mine': 0.0, 'arrest': 0.0, 'fool': 0.0, 'motley': 0.0,
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

def _update_vader_lexicon():
    """Update VADER's lexicon with financial terms"""
    if VADER_AVAILABLE:
        # Update VADER's lexicon with our financial terms
        _vader_analyzer.lexicon.update(FINANCIAL_LEXICON)
        print("✅ VADER enhanced with financial lexicon")

# Initialize enhanced VADER
if VADER_AVAILABLE:
    _update_vader_lexicon()

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

# Cache hybrid analyzer instances
@lru_cache(maxsize=1)
def get_hybrid_analyzers():
    """Get cached VADER and DistilRoBERTa analyzers"""
    vader_analyzer = get_vader_analyzer()
    ai_model, ai_tokenizer = get_ai_analyzer()
    return vader_analyzer, ai_model, ai_tokenizer

@lru_cache(maxsize=1)
def get_vader_analyzer():
    """Get cached VADER analyzer"""
    if VADER_AVAILABLE:
        return _vader_analyzer
    return None

@lru_cache(maxsize=1)
def get_ai_analyzer():
    """Get cached DistilRoBERTa analyzer with lazy initialization"""
    if _ensure_model():
        return _ai_model, _ai_tokenizer
    return None, None

@lru_cache(maxsize=1000)
def get_hybrid_sentiment(text):
    """Hybrid sentiment analysis: DistilRoBERTa for accuracy + VADER for nuanced scoring"""
    vader_analyzer, ai_model, ai_tokenizer = get_hybrid_analyzers()
    
    # Clean text for analysis
    cleaned_text = clean_headline(text)
    
    # Get VADER score (nuanced -1 to 1 range)
    vader_score = 0.0
    if vader_analyzer is not None:
        try:
            vader_scores = vader_analyzer.polarity_scores(cleaned_text)
            vader_score = vader_scores['compound']
        except Exception as e:
            print(f"VADER error: {e}")
            vader_score = get_simple_sentiment(text)
    else:
        vader_score = get_simple_sentiment(text)
    
    # Get DistilRoBERTa classification (for accuracy)
    ai_classification = None
    ai_confidence = 0.5
    
    if ai_model is not None and ai_tokenizer is not None:
        try:
            import torch
            
            # Tokenize and get model prediction
            inputs = ai_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = ai_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # DistilRoBERTa outputs: [Negative, Neutral, Positive]
            negative_score = predictions[0][0].item()
            neutral_score = predictions[0][1].item()
            positive_score = predictions[0][2].item()
            
            # Determine classification and confidence
            max_score = max(negative_score, neutral_score, positive_score)
            ai_confidence = max_score
            
            if positive_score == max_score:
                ai_classification = 'positive'
            elif negative_score == max_score:
                ai_classification = 'negative'
            else:
                ai_classification = 'neutral'
                
        except Exception as e:
            print(f"DistilRoBERTa error: {e}")
    
    # Hybrid scoring logic
    if ai_classification is None:
        # Fallback to pure VADER
        return vader_score
    
    # Use DistilRoBERTa classification to validate/correct VADER direction
    if ai_classification == 'positive':
        # If DistilRoBERTa says positive, ensure VADER score is positive
        if vader_score < 0:
            # VADER disagrees - use moderate positive score weighted by confidence
            hybrid_score = abs(vader_score) * ai_confidence * 0.7
        else:
            # VADER agrees - use VADER's nuanced score
            hybrid_score = vader_score
    
    elif ai_classification == 'negative':
        # If DistilRoBERTa says negative, ensure VADER score is negative
        if vader_score > 0:
            # VADER disagrees - use moderate negative score weighted by confidence
            hybrid_score = -abs(vader_score) * ai_confidence * 0.7
        else:
            # VADER agrees - use VADER's nuanced score
            hybrid_score = vader_score
    
    else:  # neutral
        # DistilRoBERTa says neutral - dampen VADER score
        if ai_confidence > 0.6:
            # High confidence neutral - strongly dampen
            hybrid_score = vader_score * 0.2
        else:
            # Low confidence neutral - moderately dampen
            hybrid_score = vader_score * 0.5
    
    return hybrid_score

def get_hybrid_sentiment_batch(texts, batch_size=16):
    """Batch process multiple texts using hybrid approach"""
    vader_analyzer, ai_model, ai_tokenizer = get_hybrid_analyzers()
    
    if ai_model is None or ai_tokenizer is None:
        # Fallback to VADER-only batch processing
        return get_vader_sentiment_batch(texts)
    
    try:
        import torch
        
        # Check for GPU and move model if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            ai_model = ai_model.to(device)
            print(f"🚀 Using GPU acceleration for hybrid analysis: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Using CPU processing for hybrid analysis")
        
        results = []
        
        # Get VADER scores for all texts (fast)
        vader_scores = []
        if vader_analyzer is not None:
            for text in texts:
                cleaned_text = clean_headline(text)
                try:
                    scores = vader_analyzer.polarity_scores(cleaned_text)
                    vader_scores.append(scores['compound'])
                except:
                    vader_scores.append(get_simple_sentiment(text))
        else:
            vader_scores = [get_simple_sentiment(text) for text in texts]
        
        # Process DistilRoBERTa in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_vader_scores = vader_scores[i:i + batch_size]
            
            # Clean batch texts
            cleaned_batch = [clean_headline(text) for text in batch_texts]
            
            # Tokenize batch
            inputs = ai_tokenizer(cleaned_batch, return_tensors="pt", truncation=True, 
                                 padding=True, max_length=512)
            
            # Move batch to GPU if available
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = ai_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process batch results with hybrid logic
            for j in range(len(batch_texts)):
                # DistilRoBERTa outputs: [Negative, Neutral, Positive]
                negative_score = predictions[j][0].item()
                neutral_score = predictions[j][1].item()
                positive_score = predictions[j][2].item()
                
                # Determine classification and confidence
                max_score = max(negative_score, neutral_score, positive_score)
                ai_confidence = max_score
                
                if positive_score == max_score:
                    ai_classification = 'positive'
                elif negative_score == max_score:
                    ai_classification = 'negative'
                else:
                    ai_classification = 'neutral'
                
                # Apply hybrid logic
                vader_score = batch_vader_scores[j]
                
                if ai_classification == 'positive':
                    if vader_score < 0:
                        hybrid_score = abs(vader_score) * ai_confidence * 0.7
                    else:
                        hybrid_score = vader_score
                
                elif ai_classification == 'negative':
                    if vader_score > 0:
                        hybrid_score = -abs(vader_score) * ai_confidence * 0.7
                    else:
                        hybrid_score = vader_score
                
                else:  # neutral
                    if ai_confidence > 0.6:
                        hybrid_score = vader_score * 0.2
                    else:
                        hybrid_score = vader_score * 0.5
                
                results.append(hybrid_score)
        
        return results
        
    except Exception as e:
        print(f"Hybrid batch error: {e}")
        return get_vader_sentiment_batch(texts)

@lru_cache(maxsize=1000)
def remove_source_tags(headline):
    """Remove source tags like [benzinga], [reuters] from headlines - optimized with pre-compiled regex"""
    if not headline:
        return headline
    
    cleaned = headline
    # Apply patterns in order: start, end, then middle
    cleaned = _SOURCE_TAG_PATTERNS[0].sub('', cleaned)  # Remove from start
    cleaned = _SOURCE_TAG_PATTERNS[1].sub('', cleaned)  # Remove from end
    cleaned = _SOURCE_TAG_PATTERNS[2].sub(' ', cleaned)  # Remove from middle, replace with space
    
    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def get_vader_sentiment_batch(texts, batch_size=None):
    """Batch process multiple texts using VADER only (fallback)"""
    analyzer = get_vader_analyzer()
    
    if analyzer is None:
        # Fallback to simple rule-based sentiment
        return [get_simple_sentiment(text) for text in texts]
    
    try:
        results = []
        
        # VADER is fast enough to process individually
        for text in texts:
            cleaned_text = clean_headline(text)
            scores = analyzer.polarity_scores(cleaned_text)
            results.append(scores['compound'])
        
        return results
        
    except Exception as e:
        print(f"VADER batch error: {e}")
        return [get_simple_sentiment(text) for text in texts]

@lru_cache(maxsize=1000)
def get_simple_sentiment(text):
    """Simple rule-based sentiment analysis as fallback"""
    text_lower = text.lower()
    
    positive_words = ['good', 'great', 'excellent', 'strong', 'beat', 'exceed', 'rally', 'soar', 'gain', 'profit', 'bullish', 'upgrade']
    negative_words = ['bad', 'poor', 'weak', 'miss', 'fall', 'drop', 'decline', 'loss', 'bearish', 'downgrade', 'crash', 'plunge']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return min(0.8, positive_count * 0.2)
    elif negative_count > positive_count:
        return max(-0.8, -negative_count * 0.2)
    else:
        return 0.0

@lru_cache(maxsize=1000)
def clean_headline(text):
    """Optimized headline cleaning with caching - enhanced for VADER"""
    if not text:
        return text
        
    text_lower = text.lower()
    
    # Financial phrase replacements for better VADER analysis
    replacements = {
        'earnings fall short': 'disappointing earnings',
        'falls short': 'disappointing', 'fell short': 'disappointing',
        'revenue tops estimates': 'strong revenue', 'tops estimates': 'beat expectations',
        'exceeds expectations': 'beat expectations', 'below estimates': 'miss expectations',
        'break losing streak': 'rally', 'snaps losing streak': 'rally',
        'recovers from': 'rally', 'bounce back': 'rally',
        'top pick': 'best stock', 'buy rating': 'strong buy', 'strong buy': 'excellent',
        'outperform': 'winner', 'stock to buy': 'good investment'
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
    """Optimized sentiment analysis with hybrid VADER + DistilRoBERTa approach"""
    if not parsed_data:
        return pd.DataFrame()

    # Create DataFrame with optimized dtypes for memory efficiency
    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert timestamps to datetime if needed (vectorized)
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    
    # Optimize memory usage by using categorical data types for repeated strings
    df['Source'] = df['Source'].astype('category')
    
    # Get cached hybrid analyzers
    vader_analyzer, ai_model, ai_tokenizer = get_hybrid_analyzers()
    
    # Vectorized headline cleaning
    df['Cleaned_Headline'] = df['Headline'].apply(clean_headline)
    
    # Print simple summary (no source breakdown)
    print(f"Analyzing {len(df)} news articles...")
    
    # Add progress bar for sentiment analysis
    try:
        from tqdm import tqdm
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
    
    # Hybrid sentiment analysis with progress bar
    if ai_model is not None and ai_tokenizer is not None and vader_analyzer is not None:
        print("Using hybrid VADER + DistilRoBERTa analysis...")
        
        # Use batch processing for optimal performance
        headlines_list = df['Cleaned_Headline'].tolist()
        
        if use_progress_bar:
            print("Processing headlines with hybrid approach...")
            raw_scores = get_hybrid_sentiment_batch(headlines_list, batch_size=16)
        else:
            raw_scores = get_hybrid_sentiment_batch(headlines_list, batch_size=16)
            
        raw_scores = np.array(raw_scores)
    elif vader_analyzer is not None:
        print("Using VADER with enhanced financial lexicon...")
        headlines_list = df['Cleaned_Headline'].tolist()
        raw_scores = get_vader_sentiment_batch(headlines_list)
        raw_scores = np.array(raw_scores)
    else:
        print("Using fallback sentiment analysis...")
        if use_progress_bar:
            raw_scores = np.array([get_simple_sentiment(headline) 
                                 for headline in tqdm(df['Cleaned_Headline'], desc="Processing headlines")])
        else:
            raw_scores = np.array([get_simple_sentiment(headline) for headline in df['Cleaned_Headline']])
    
    df['Raw_Score'] = raw_scores
    
    # Vectorized target validation using numpy where for better performance
    valid_mask = df.apply(lambda row: validate_target(row['Headline'], ticker, row['Raw_Score']), axis=1)
    df['Raw_Score'] = valid_mask
    
    # Optimized source categorization with vectorized operations
    df['Source_Type'] = df['Source'].apply(categorize_source).astype('category')
    
    # Vectorized weight calculation using numpy for better performance
    source_weights = df['Source'].map(lambda s: SOURCE_WEIGHTS.get(s.title(), SOURCE_WEIGHTS.get(s, 1.0)))
    df['Weight'] = source_weights.astype('float32')  # Use float32 for memory efficiency
    df['Compound_Score'] = np.clip(df['Raw_Score'] * df['Weight'], -1.0, 1.0).astype('float32')
    
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
    
    # Clean up intermediate DataFrame to free memory
    del df
    
    return result_df

@lru_cache(maxsize=1000)
def get_top_headlines(parsed_data_tuple, ticker, timeframe_days=30):
    """Optimized top headlines extraction with hybrid sentiment analysis"""
    if not parsed_data_tuple:
        return "N/A", "N/A"

    # Convert tuple back to list for DataFrame creation
    parsed_data = list(parsed_data_tuple)
    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    if df.empty:
        return "No significant news.", "No significant news."
    
    # Get cached hybrid analyzers
    vader_analyzer, ai_model, ai_tokenizer = get_hybrid_analyzers()
    
    # Vectorized operations - use clean_headline function
    df['Cleaned'] = df['Headline'].apply(clean_headline)
    
    if ai_model is not None and ai_tokenizer is not None and vader_analyzer is not None:
        # Use hybrid batch processing for better performance
        cleaned_headlines = df['Cleaned'].tolist()
        scores = get_hybrid_sentiment_batch(cleaned_headlines, batch_size=16)
        df['Score'] = scores
    elif vader_analyzer is not None:
        # Use VADER only
        cleaned_headlines = df['Cleaned'].tolist()
        scores = get_vader_sentiment_batch(cleaned_headlines)
        df['Score'] = scores
    else:
        # Fallback to simple sentiment
        df['Score'] = df['Cleaned'].apply(get_simple_sentiment)
        
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