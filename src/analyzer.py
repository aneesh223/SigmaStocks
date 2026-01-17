import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Updated source categorization based on reliability and purpose
# 1. THE HOLY GRAIL - Unvarnished facts, no opinions, just numbers
PRIMARY_SOURCES = {
    'SEC EDGAR', 'SEC', 'U.S. Securities and Exchange Commission', 
    'Company Investor Relations', 'Investor Relations'
}

# 2. THE PROFESSIONAL GRADE - High integrity, expensive, fact-checked
# Used by people moving millions, not hundreds
INSTITUTIONAL_SOURCES = {
    'Bloomberg', 'Reuters', 'The Wall Street Journal', 'Financial Times', 
    'Barron\'s', 'Morningstar', 'Investor\'s Business Daily'
}

# 3. THE AGGREGATORS - Good for quick data checks, mix hard data with opinions
DATA_AGGREGATORS = {
    'Yahoo Finance', 'MarketWatch', 'Seeking Alpha', 'Zacks', 'TipRanks'
}

# 4. THE NOISE MACHINE - Exists to sell ads, excitement, and anxiety
ENTERTAINMENT_SOURCES = {
    'The Motley Fool', 'CNBC', 'CNN Business', 'Fox Business', 'MSN',
    'USA Today', 'Forbes', 'Fortune', 'Business Insider', 'Reddit',
    'Twitter', 'StockTwits', 'Robinhood Snacks', 'Benzinga', 'TheStreet'
}

# Updated weights based on new categorization
SOURCE_WEIGHTS = {
    # Primary sources get highest weight
    'SEC EDGAR': 2.0, 'SEC': 2.0, 'U.S. Securities and Exchange Commission': 2.0,
    'Company Investor Relations': 2.0, 'Investor Relations': 2.0,
    
    # Institutional sources - high weight
    'Bloomberg': 1.8, 'Reuters': 1.8, 'The Wall Street Journal': 1.8, 
    'Financial Times': 1.8, 'Barron\'s': 1.7, 'Morningstar': 1.6,
    'Investor\'s Business Daily': 1.6,
    
    # Data aggregators - moderate weight
    'Yahoo Finance': 1.2, 'MarketWatch': 1.2, 'Seeking Alpha': 1.0, 
    'Zacks': 1.0, 'TipRanks': 1.0,
    
    # Entertainment sources - lower weight
    'The Motley Fool': 0.6, 'CNBC': 0.8, 'CNN Business': 0.7, 
    'Fox Business': 0.7, 'MSN': 0.6, 'USA Today': 0.5, 'Forbes': 0.7,
    'Fortune': 0.7, 'Business Insider': 0.6, 'Reddit': 0.4, 'Twitter': 0.3,
    'StockTwits': 0.4, 'Robinhood Snacks': 0.3, 'Benzinga': 0.5, 'TheStreet': 0.5
}

def clean_headline(text):
    text = text.lower()
    
    replacements = {
        "break losing streak": "rally",
        "break brutal losing streak": "major rally",
        "snaps losing streak": "rally",
        "ends losing streak": "rally",
        "recovers from": "rally",
        "bounce back": "rally",
        "remains top pick": "best stock",
        "top pick": "best stock",
        "averts disaster": "success",
        "avoids crash": "success",
        "not as bad": "better",
        "less bad": "improving",
        "buy rating": "perfect",
        "strong buy": "perfect",
        "outperform": "winner",
        "if i could only buy": "best stock",
        "single stock": "best stock",
        "only buy and hold": "perfect stock",
        "stock to buy": "good stock"
    }
    
    for phrase, replacement in replacements.items():
        if phrase in text:
            text = text.replace(phrase, replacement)
            
    return text

def validate_target(headline, ticker, raw_score):
    headline_lower = headline.lower()
    ticker_lower = ticker.lower()
    
    if " for " in headline_lower:
        parts = headline_lower.split(" for ")
        if len(parts) > 1:
            target = parts[1].split()[0]
            target = target.strip(".,;:!?")
            
            generics = ['investors', 'shareholders', 'holders', 'stock', 'market', 
                        'trading', 'tech', 'semis', 'sector', 'growth', 'economy']
            
            if target != ticker_lower and target not in generics:
                return 0.0
                    
    return raw_score

def categorize_source(source):
    """Categorize news source based on reliability and purpose hierarchy"""
    # Check exact matches first
    if source in PRIMARY_SOURCES:
        return 'primary'
    elif source in INSTITUTIONAL_SOURCES:
        return 'institutional'
    elif source in DATA_AGGREGATORS:
        return 'aggregator'
    elif source in ENTERTAINMENT_SOURCES:
        return 'entertainment'
    else:
        # Smart categorization based on keywords for unknown sources
        source_lower = source.lower()
        
        # Primary source keywords
        if any(keyword in source_lower for keyword in ['sec', 'securities and exchange commission', 'investor relations']):
            return 'primary'
        
        # Institutional keywords
        elif any(keyword in source_lower for keyword in ['bloomberg', 'reuters', 'wall street journal', 'financial times', 'barron']):
            return 'institutional'
        
        # Aggregator keywords
        elif any(keyword in source_lower for keyword in ['yahoo finance', 'marketwatch', 'seeking alpha', 'zacks']):
            return 'aggregator'
        
        # Entertainment keywords
        elif any(keyword in source_lower for keyword in ['motley', 'fool', 'reddit', 'twitter', 'robinhood', 'benzinga', 'thestreet']):
            return 'entertainment'
        
        # Default to aggregator for unknown financial sources
        elif any(keyword in source_lower for keyword in ['financial', 'market', 'invest', 'stock', 'trading']):
            return 'aggregator'
        
        else:
            return 'entertainment'  # Default to lowest tier for unknown sources

def get_sentiment(parsed_data, ticker, timeframe_days=30):
    if not parsed_data:
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    vader = sia()
    
    financial_lexicon = {
        'error': 0.0, 'loom': 0.0, 'vice': 0.0, 'tank': -2.5,
        'gross': 0.0, 'mine': 0.0, 'arrest': 0.0,
        'fool': 0.0, 'motley': 0.0, 
        'beat': 2.5, 'miss': -2.5, 'crush': 3.0, 'surprise': 1.5,
        'plunge': -3.0, 'soar': 3.0, 'skyrocket': 3.5, 
        'rally': 2.5, 'brutal': -3.0,
        'flat': -1.5, 'stagnant': -1.5, 'choppy': -1.0, 'sideways': -1.0,
        'shorting': -2.0, 'call': 1.0, 'put': -1.0
    }
    vader.lexicon.update(financial_lexicon)
    
    df['Cleaned_Headline'] = df['Headline'].apply(clean_headline)
    
    df['Raw_Score'] = df['Cleaned_Headline'].apply(lambda title: vader.polarity_scores(title)['compound'])
    
    df['Raw_Score'] = df.apply(lambda row: validate_target(row['Headline'], ticker, row['Raw_Score']), axis=1)
    
    # Categorize sources
    df['Source_Type'] = df['Source'].apply(categorize_source)
    
    df['Weight'] = df['Source'].apply(lambda x: SOURCE_WEIGHTS.get(x, 1.0))
    df['Compound_Score'] = df['Raw_Score'] * df['Weight']
    
    df['Compound_Score'] = df['Compound_Score'].clip(-1.0, 1.0)
    
    if timeframe_days <= 1:
        df['TimeGroup'] = df['Timestamp'].dt.floor('h')
    else:
        df['TimeGroup'] = df['Timestamp'].dt.date
    
    # Calculate separate sentiment scores by source type
    primary_df = df[df['Source_Type'] == 'primary']
    institutional_df = df[df['Source_Type'] == 'institutional']
    aggregator_df = df[df['Source_Type'] == 'aggregator']
    entertainment_df = df[df['Source_Type'] == 'entertainment']
    
    # Group by time and calculate means for each category
    overall_scores = df.groupby(['TimeGroup']).agg({
        'Compound_Score': 'mean',
        'Raw_Score': 'mean',
        'Weight': 'mean'
    })
    
    primary_scores = primary_df.groupby(['TimeGroup'])['Compound_Score'].mean() if not primary_df.empty else pd.Series(dtype=float)
    institutional_scores = institutional_df.groupby(['TimeGroup'])['Compound_Score'].mean() if not institutional_df.empty else pd.Series(dtype=float)
    aggregator_scores = aggregator_df.groupby(['TimeGroup'])['Compound_Score'].mean() if not aggregator_df.empty else pd.Series(dtype=float)
    entertainment_scores = entertainment_df.groupby(['TimeGroup'])['Compound_Score'].mean() if not entertainment_df.empty else pd.Series(dtype=float)
    
    # Combine all sentiment data
    result_df = overall_scores.copy()
    result_df['Primary_Sentiment'] = primary_scores
    result_df['Institutional_Sentiment'] = institutional_scores
    result_df['Aggregator_Sentiment'] = aggregator_scores
    result_df['Entertainment_Sentiment'] = entertainment_scores
    
    # Fill NaN values with 0 for missing sentiment types
    result_df['Primary_Sentiment'] = result_df['Primary_Sentiment'].fillna(0)
    result_df['Institutional_Sentiment'] = result_df['Institutional_Sentiment'].fillna(0)
    result_df['Aggregator_Sentiment'] = result_df['Aggregator_Sentiment'].fillna(0)
    result_df['Entertainment_Sentiment'] = result_df['Entertainment_Sentiment'].fillna(0)
    
    # Calculate counts for each category
    primary_counts = primary_df.groupby(['TimeGroup']).size() if not primary_df.empty else pd.Series(dtype=int)
    institutional_counts = institutional_df.groupby(['TimeGroup']).size() if not institutional_df.empty else pd.Series(dtype=int)
    aggregator_counts = aggregator_df.groupby(['TimeGroup']).size() if not aggregator_df.empty else pd.Series(dtype=int)
    entertainment_counts = entertainment_df.groupby(['TimeGroup']).size() if not entertainment_df.empty else pd.Series(dtype=int)
    
    result_df['Primary_Count'] = primary_counts.reindex(result_df.index, fill_value=0)
    result_df['Institutional_Count'] = institutional_counts.reindex(result_df.index, fill_value=0)
    result_df['Aggregator_Count'] = aggregator_counts.reindex(result_df.index, fill_value=0)
    result_df['Entertainment_Count'] = entertainment_counts.reindex(result_df.index, fill_value=0)
    
    print(f"Source breakdown - Primary: {len(primary_df)}, Institutional: {len(institutional_df)}, Aggregator: {len(aggregator_df)}, Entertainment: {len(entertainment_df)}")
    
    return result_df

def get_top_headlines(parsed_data, ticker, timeframe_days=30):
    if not parsed_data:
        return "N/A", "N/A"

    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    vader = sia()
    vader.lexicon.update({'beat': 2.5, 'miss': -2.5, 'rally': 2.5, 'flat': -1.5, 'fool': 0.0})

    df['Cleaned'] = df['Headline'].apply(clean_headline)
    df['Score'] = df['Cleaned'].apply(lambda title: vader.polarity_scores(title)['compound'])
    df['Score'] = df.apply(lambda row: validate_target(row['Headline'], ticker, row['Score']), axis=1)

    df = df.sort_values(by='Score', ascending=False)
    
    if df.empty:
        return "No significant news.", "No significant news."

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