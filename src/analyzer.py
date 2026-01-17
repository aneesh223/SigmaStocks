import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

# Download VADER lexicon if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- 1. SOURCE RELIABILITY WEIGHTS ---
# Acts as a "Volume Knob" for credibility.
SOURCE_WEIGHTS = {
    'Bloomberg': 1.5,
    'Reuters': 1.5,
    'The Wall Street Journal': 1.5,
    'Financial Times': 1.5,
    'CNBC': 1.2,
    'Yahoo Finance': 1.0,
    'MarketWatch': 1.0,
    'The Motley Fool': 0.7,   # Often clickbait/opinion
    'Seeking Alpha': 0.7,     # User-generated content
    'Benzinga': 0.8,
    'Zacks': 0.8,
    'MSN': 1.0
}

def clean_headline(text):
    """
    Replaces complex idioms that confuse VADER with simple emotional words.
    Example: "Break losing streak" -> "Major Rally"
    """
    text = text.lower()
    
    # --- PHRASE MAPPINGS ---
    replacements = {
        # Breaking a bad trend -> GOOD
        "break losing streak": "rally",
        "break brutal losing streak": "major rally",
        "snaps losing streak": "rally",
        "ends losing streak": "rally",
        "recovers from": "rally",
        "bounce back": "rally",
        "remains top pick": "best stock",
        "top pick": "best stock",
        
        # Preventing a bad thing -> GOOD
        "averts disaster": "success",
        "avoids crash": "success",
        
        # "Not bad" -> GOOD
        "not as bad": "better",
        "less bad": "improving",
        
        # "Top Pick" logic
        "buy rating": "perfect",
        "strong buy": "perfect",
        "outperform": "winner",
        
        # Clickbait Fixes (NEW)
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
    """
    Checks if the sentiment is actually meant for a DIFFERENT company.
    Example: "Bad news for Tesla" (When analyzing NVDA).
    """
    headline_lower = headline.lower()
    ticker_lower = ticker.lower()
    
    # Logic: If we see "for [Target]", check if Target is us.
    if " for " in headline_lower:
        parts = headline_lower.split(" for ")
        if len(parts) > 1:
            # Get the word immediately following "for"
            target = parts[1].split()[0]
            
            # Remove punctuation (e.g., "tesla," -> "tesla")
            target = target.strip(".,;:!?")
            
            # Words that are generic and usually apply to us
            generics = ['investors', 'shareholders', 'holders', 'stock', 'market', 
                        'trading', 'tech', 'semis', 'sector', 'growth', 'economy']
            
            # If target is NOT our ticker AND NOT a generic word...
            # Then this news is likely for a competitor.
            if target != ticker_lower and target not in generics:
                # If the score is negative/positive, we neutralize it because it's not for us.
                # (e.g. "Bad news for Tesla" shouldn't hurt NVDA's score)
                return 0.0
                    
    return raw_score

def get_sentiment(parsed_data, ticker):
    """
    Main Logic: Clean -> Score -> Validate -> Weight
    """
    if not parsed_data:
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    # Initialize VADER
    vader = sia()
    
    # --- 2. PATCH THE DICTIONARY ---
    # Teach VADER financial slang
    financial_lexicon = {
        # False Negatives (VADER thinks bad, Finance thinks neutral/good)
        'error': 0.0, 'loom': 0.0, 'vice': 0.0, 'tank': -2.5,
        'gross': 0.0, 'mine': 0.0, 'arrest': 0.0,
        
        # Brand Names (NEW)
        'fool': 0.0, 'motley': 0.0, 
        
        # Earnings
        'beat': 2.5, 'miss': -2.5, 'crush': 3.0, 'surprise': 1.5,
        
        # Movement
        'plunge': -3.0, 'soar': 3.0, 'skyrocket': 3.5, 
        'rally': 2.5, 'brutal': -3.0,
        
        # Boring/Bad Conditions (The "Flat" Fix)
        'flat': -1.5, 'stagnant': -1.5, 'choppy': -1.0, 'sideways': -1.0,
        
        # Options/Trading
        'shorting': -2.0, 'call': 1.0, 'put': -1.0
    }
    vader.lexicon.update(financial_lexicon)
    
    # Step A: Clean Headlines (Translate idioms)
    df['Cleaned_Headline'] = df['Headline'].apply(clean_headline)
    
    # Step B: Calculate Raw Score
    df['Raw_Score'] = df['Cleaned_Headline'].apply(lambda title: vader.polarity_scores(title)['compound'])
    
    # Step C: Validate Target (Ignore "For Competitor")
    df['Raw_Score'] = df.apply(lambda row: validate_target(row['Headline'], ticker, row['Raw_Score']), axis=1)
    
    # Step D: Apply Source Weights
    df['Weight'] = df['Source'].apply(lambda x: SOURCE_WEIGHTS.get(x, 1.0))
    df['Compound_Score'] = df['Raw_Score'] * df['Weight']
    
    # Clamp score to stay within -1.0 to 1.0
    df['Compound_Score'] = df['Compound_Score'].clip(-1.0, 1.0)
    
    # Group by Date
    df['Date'] = df['Timestamp'].dt.date
    mean_scores = df.groupby(['Date']).mean(numeric_only=True)
    
    return mean_scores

def get_top_headlines(parsed_data, ticker):
    """
    Returns the most impactful headlines for display.
    """
    if not parsed_data:
        return "N/A", "N/A"

    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    # Re-initialize VADER for ranking
    vader = sia()
    vader.lexicon.update({'beat': 2.5, 'miss': -2.5, 'rally': 2.5, 'flat': -1.5, 'fool': 0.0})

    # Apply same logic chain for ranking
    df['Cleaned'] = df['Headline'].apply(clean_headline)
    df['Score'] = df['Cleaned'].apply(lambda title: vader.polarity_scores(title)['compound'])
    df['Score'] = df.apply(lambda row: validate_target(row['Headline'], ticker, row['Score']), axis=1)

    df = df.sort_values(by='Score', ascending=False)
    
    if df.empty:
        return "No significant news.", "No significant news."

    top_story = df.iloc[0]
    bottom_story = df.iloc[-1]
    
    # Format Output
    if top_story['Score'] > 0.1:
        best_headline = f"[{top_story['Source']}] {top_story['Headline']} (Score: {top_story['Score']:.2f})"
    else:
        best_headline = "No significant positive news found."

    if bottom_story['Score'] < -0.1:
        worst_headline = f"[{bottom_story['Source']}] {bottom_story['Headline']} (Score: {bottom_story['Score']:.2f})"
    else:
        worst_headline = "No significant negative news found."
    
    return best_headline, worst_headline