import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def get_sentiment(parsed_data):
    df = pd.DataFrame(parsed_data, columns=['Date', 'Time', 'Headline'])
    
    vader = sia()
    df['Compound_Score'] = df['Headline'].apply(lambda title: vader.polarity_scores(title)['compound'])
    
    df['Date'] = pd.to_datetime(df['Date'], format='%b-%d-%y', errors='coerce').dt.date

    mean_scores = df.groupby(['Date']).mean(numeric_only=True)
    
    return mean_scores