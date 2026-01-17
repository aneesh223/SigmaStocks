from datetime import datetime, timedelta
import scraper
import analyzer
import market
import visualizer

def analyze_stock(ticker, lookback_days):
    """Analyze a single stock with given parameters"""
    print(f"\nFetching data for {ticker} over last {lookback_days} days...")
    
    try:
        # 3. SCRAPE DATA (Dynamic Timeframe)
        # We tell the scraper exactly how many days we need
        profile_data, raw_data = scraper.scrape_finviz(ticker, days=lookback_days)
        
        # Unpack profile
        company_name, industry, mkt_cap = profile_data
        
    except Exception as e:
        print(f"Error scraping data: {e}")
        return
    
    if raw_data:
        # 4. ANALYZE SENTIMENT (AI Mode)
        print("Analyzing sentiment... (AI running)")
        
        # Pass 'ticker' so the AI knows which company to focus on (Context Filter)
        sentiment_df = analyzer.get_sentiment(raw_data, ticker, lookback_days)
        best_news, worst_news = analyzer.get_top_headlines(raw_data, ticker, lookback_days)
        
        # 5. FILTER DATES LOCALLY
        # Ensure we don't accidentally include data outside the requested window
        today = datetime.now()
        cutoff_date = (today - timedelta(days=lookback_days)).date()
        
        # Handle different index types for comparison
        if lookback_days <= 1:
            # For hourly data, convert cutoff to datetime for comparison
            import pytz
            cutoff_datetime = datetime.combine(cutoff_date, datetime.min.time()).replace(tzinfo=pytz.utc)
            sentiment_df = sentiment_df[sentiment_df.index >= cutoff_datetime]
        else:
            # For daily data, ensure index is date type for comparison
            if hasattr(sentiment_df.index[0], 'date'):
                sentiment_df.index = sentiment_df.index.date
            sentiment_df = sentiment_df[sentiment_df.index >= cutoff_date]

        if sentiment_df.empty:
            print(f"No relevant news found for {ticker} in the last {lookback_days} days.")
        else:
            # 6. MARKET DATA & ALGO
            final_df = market.get_financials(ticker, sentiment_df, lookback_days)
            
            # 7. PRINT VERDICT
            # This now uses the "Weighted Average" logic (Recency Bias)
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
            
            # 8. PLOT GRAPH
            visualizer.plot_graph(ticker, final_df, profile_data, lookback_days)
    else:
        print("Scraping failed or no data found.")

def main():
    print("--------------------------------------------------")
    print("   STOCK IQ: AI SENTIMENT & VALUE ANALYZER")
    print("--------------------------------------------------")
    print("Enter 'EXIT' to stop the program")
    print("--------------------------------------------------")
    
    while True:
        # 1. Inputs
        ticker = input("\nEnter ticker (e.g., TSLA) or EXIT to exit the program: ").upper().strip()
        
        # Check for exit command
        if ticker == 'EXIT':
            print("Thank you for using StockIQ! Goodbye!")
            break

        if not ticker:
            print("Please enter a valid ticker symbol.")
            continue

        print("\nSelect Timeframe:")
        print("1. 1D  (1 Day)")
        print("2. 5D  (5 Days)")
        print("3. 1M  (1 Month)")
        print("4. 6M  (6 Months)")
        print("5. YTD (Year-To-Date)")
        print("6. MAX (Max Available Data)") # Combined 1Y/MAX
        
        choice = input("Enter choice (e.g., 5D, 2): ").upper().strip()

        # --- 2. CALCULATE LOOKBACK DAYS ---
        today = datetime.now()
        
        if choice in ['1', '1D']:
            lookback_days = 1
        elif choice in ['2', '5D']:
            lookback_days = 5
        elif choice in ['3', '1M']:
            lookback_days = 30
        elif choice in ['4', '6M']:
            lookback_days = 180
        elif choice in ['5', 'YTD']:
            # Calculate days since Jan 1st
            start_of_year = datetime(today.year, 1, 1)
            delta = today - start_of_year
            lookback_days = delta.days
        elif choice in ['6', 'MAX', '1Y']:
            # We cap at 365 because Google News rarely provides reliable data older than that
            lookback_days = 365 
        else:
            print("Invalid selection. Defaulting to 1 Month.")
            lookback_days = 30

        # Analyze the stock
        analyze_stock(ticker, lookback_days)

if __name__ == "__main__":
    main()