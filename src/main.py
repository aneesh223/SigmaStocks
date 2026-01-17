from datetime import datetime, timedelta
import scraper
import analyzer
import market
import visualizer

def main():
    ticker = input("Enter ticker (e.g., TSLA): ").upper()

    print("\nSelect Industry Standard Timeframe:")
    print("1. 1D  (1 Day)")
    print("2. 5D  (5 Days)")
    print("3. 1W  (1 Week)")
    print("4. 1M  (1 Month)")
    print("5. YTD (Year-To-Date)")
    print("6. 1Y  (1 Year)")
    print("7. MAX (All Available Data)")
    
    choice = input("Enter choice (e.g., 5D, 2): ").upper()

    print(f"Fetching data for {ticker}...")
    try:
        profile_data, raw_data = scraper.scrape_finviz(ticker)
        company_name, industry, mkt_cap = profile_data
    except Exception as e:
        print(f"Error scraping data: {e}")
        return
    
    if raw_data:
        sentiment_df = analyzer.get_sentiment(raw_data)
        
        # --- NEW: GET EVIDENCE HEADLINES ---
        # We look at ALL data to find the most impactful headlines
        best_news, worst_news = analyzer.get_top_headlines(raw_data)
        # -----------------------------------
        
        # Filter Dates
        today = datetime.now().date()
        cutoff_date = None
        
        if choice in ['1', '1D']:
            cutoff_date = today - timedelta(days=1)
        elif choice in ['2', '5D']:
            cutoff_date = today - timedelta(days=5)
        elif choice in ['3', '1W']:
            cutoff_date = today - timedelta(days=7)
        elif choice in ['4', '1M']:
            cutoff_date = today - timedelta(days=30)
        elif choice in ['5', 'YTD']:
            cutoff_date = datetime(today.year, 1, 1).date()
        elif choice in ['6', '1Y']:
            cutoff_date = today - timedelta(days=365)
        elif choice in ['7', 'MAX']:
            cutoff_date = None
        else:
            cutoff_date = None
            
        if cutoff_date:
            print(f"Filtering data since: {cutoff_date}")
            sentiment_df = sentiment_df[sentiment_df.index >= cutoff_date]

        if sentiment_df.empty:
            print("No news found in this date range.")
        else:
            final_df = market.get_financials(ticker, sentiment_df)
            verdict = market.calculate_verdict(final_df)
            
            if verdict:
                print("\n" + "="*50)
                print(f"   ANALYSIS FOR {company_name} ({ticker})")
                print("="*50)
                print(f"Industry:   {industry}")
                print(f"Market Cap: {mkt_cap}")
                print("-" * 50)
                print(f"Sentiment Health: {verdict['Sentiment_Score']}/10")
                print(f"Price Value:      {verdict['Value_Score']}/10")
                print("-" * 50)
                print(f"FINAL BUY SCORE:  {verdict['Final_Score']}/10")
                print(f"\nREASONING:\n{verdict['Explanation']}")
                print("-" * 50)
                print(f"TOP POSITIVE NEWS:\n>> {best_news}")
                print(f"\nTOP NEGATIVE NEWS:\n>> {worst_news}")
                print("="*50 + "\n")
            
            visualizer.plot_graph(ticker, final_df, profile_data)
    else:
        print("Scraping failed or no data found.")

if __name__ == "__main__":
    main()