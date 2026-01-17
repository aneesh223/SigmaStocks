import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

# Download VADER lexicon if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def get_sentiment(parsed_data):
    """
    Converts raw list to DataFrame and calculates Compound Sentiment Score.
    """
    if not parsed_data:
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline'])
    
    # 1. Initialize VADER
    vader = sia()
    
    # --- 2. PATCH THE DICTIONARY (THE FIX) ---
    # We manually override words that confuse VADER in a financial context.
    # Scores range from -4.0 (Most Negative) to +4.0 (Most Positive).
    # --- 2. PATCH THE DICTIONARY (THE FIX) ---
    financial_lexicon = {
        # --- FALSE NEGATIVES (Words VADER thinks are bad, but are neutral/good in finance) ---
        'error': 0.0,       # "Margin for error" -> Neutral
        'loom': 0.0,        # "Earnings loom" -> Neutral
        'vice': 0.0,        # "Vice President" -> Neutral (VADER thinks it's a sin)
        'tank': -2.5,       # "Stock tanked" -> Bad (VADER thinks it's a war tank)
        'arrest': 0.0,      # "Arrested decline" -> Neutral/Good (VADER thinks police arrest)
        'gross': 0.0,       # "Gross Domestic Product" -> Neutral (VADER thinks 'disgusting')
        'mine': 0.0,        # "Data mining" / "Gold mine" -> Neutral (VADER thinks landmine)
        'liability': -1.5,  # "Liability" -> Debt (VADER might miss the financial weight)
        
        # --- EARNINGS JARGON ---
        'beat': 2.5,        # "Beat estimates" -> Very Good
        'miss': -2.5,       # "Missed estimates" -> Very Bad
        'crush': 3.0,       # "Crushed expectations" -> Amazing
        'surprise': 1.5,    # "Earnings surprise" -> Good
        'guidance': 0.5,    # "Positive guidance" -> Good context
        'slump': -2.5,      # "Sales slump" -> Bad
        
        # --- MARKET MOVEMENT ---
        'bull': 2.0,        # "Bull market" -> Good
        'bear': -2.0,       # "Bear market" -> Bad
        'rally': 2.5,       # "Stock rally" -> Good
        'plunge': -3.0,     # "Stock plunge" -> Severe drop
        'soar': 3.0,        # "Stock soar" -> Severe rise
        'skyrocket': 3.5,   # "Skyrocket" -> Huge rise
        'correction': -1.5, # "Market correction" -> Moderate drop
        'crash': -3.5,      # "Market crash" -> Disaster
        'rebound': 2.0,     # "Stock rebound" -> Good
        'volatile': -1.0,   # "Volatile trading" -> Bad (Risk)
        
        # --- CORPORATE ACTIONS ---
        'dividend': 1.5,    # "Pays dividend" -> Good
        'buyback': 2.0,     # "Stock buyback" -> Good
        'layoff': -2.0,     # "Layoffs" -> Bad (usually)
        'hiring': 1.5,      # "Hiring spree" -> Good
        'upgrade': 2.5,     # "Analyst upgrade" -> Good
        'downgrade': -2.5,  # "Analyst downgrade" -> Bad
        'default': -3.5,    # "Debt default" -> Bankruptcy risk
        'bankrupt': -3.8,   # "Bankruptcy" -> Game over
        
        # --- ECONOMIC MACRO ---
        'inflation': -1.5,  # "High inflation" -> Bad for markets
        'recession': -3.0,  # "Recession fears" -> Bad
        'hike': -1.5,       # "Rate hike" -> Bad (usually tighter money)
        'cut': 1.5,         # "Rate cut" -> Good (cheaper money)
        'hawk': -1.0,       # "Hawkish Fed" -> Tighter money (Bad)
        'dove': 1.0,        # "Dovish Fed" -> Looser money (Good)
        'stimulus': 2.0     # "Stimulus check" -> Good
    }
    
    # Update the internal dictionary
    vader.lexicon.update(financial_lexicon)
    # -----------------------------------------
    
    # Apply Score
    df['Compound_Score'] = df['Headline'].apply(lambda title: vader.polarity_scores(title)['compound'])
    
    # Create a 'Date' column for daily grouping
    df['Date'] = df['Timestamp'].dt.date

    # Group by Date
    mean_scores = df.groupby(['Date']).mean(numeric_only=True)
    
    return mean_scores

def get_top_headlines(parsed_data):
    """
    Finds the single most positive and most negative headline in the dataset.
    """
    if not parsed_data:
        return "N/A", "N/A"

    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline'])
    
    # Re-initialize VADER with the same patches for consistency
    vader = sia()
    financial_lexicon = {
        'error': 0.0, 'loom': 0.0, 'miss': -1.5, 'beat': 1.5,
        'crush': 2.0, 'plunge': -2.5, 'soar': 2.5
    }
    vader.lexicon.update(financial_lexicon)

    df['Score'] = df['Headline'].apply(lambda title: vader.polarity_scores(title)['compound'])
    
    df = df.sort_values(by='Score', ascending=False)
    
    if df.empty:
        return "No news found.", "No news found."

    top_story = df.iloc[0]
    bottom_story = df.iloc[-1]
    
    # Logic: Only show if it's actually positive/negative
    if top_story['Score'] > 0.1:
        best_headline = f"{top_story['Headline']} (Score: {top_story['Score']})"
    else:
        best_headline = "No significant positive news found."

    if bottom_story['Score'] < -0.1:
        worst_headline = f"{bottom_story['Headline']} (Score: {bottom_story['Score']})"
    else:
        worst_headline = "No significant negative news found."
    
    return best_headline, worst_headline