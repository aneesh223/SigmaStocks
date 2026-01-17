from datetime import datetime, timedelta
import scraper
import analyzer
import market
import visualizer

def analyze_stock(ticker, lookback_days, strategy_mode):
    print(f"\nFetching data for {ticker} over last {lookback_days} days...")
    print(f"Strategy: {strategy_mode}")
    
    try:
        profile_data, raw_data = scraper.scrape_gnews(ticker, days=lookback_days)
        company_name, industry, mkt_cap = profile_data
        
    except Exception as e:
        print(f"Error scraping data: {e}")
        return
    
    if raw_data:
        print("Analyzing sentiment... (AI running)")
        
        sentiment_df = analyzer.get_sentiment(raw_data, ticker, lookback_days)
        best_news, worst_news = analyzer.get_top_headlines(raw_data, ticker, lookback_days)
        
        today = datetime.now()
        cutoff_date = (today - timedelta(days=lookback_days)).date()
        
        if lookback_days <= 1:
            import pytz
            cutoff_datetime = datetime.combine(cutoff_date, datetime.min.time()).replace(tzinfo=pytz.utc)
            sentiment_df = sentiment_df[sentiment_df.index >= cutoff_datetime]
        else:
            if hasattr(sentiment_df.index[0], 'date'):
                sentiment_df.index = sentiment_df.index.date
            sentiment_df = sentiment_df[sentiment_df.index >= cutoff_date]

        if sentiment_df.empty:
            print(f"No relevant news found for {ticker} in the last {lookback_days} days.")
        else:
            final_df = market.get_financials(ticker, sentiment_df, lookback_days)
            
            verdict = market.calculate_verdict(final_df, strategy_mode)
            
            if verdict:
                print("\n" + "="*50)
                print(f"   ANALYSIS FOR {company_name} ({ticker})")
                print("="*50)
                print(f"Industry:   {industry}")
                print(f"Market Cap: {mkt_cap}")
                print("-" * 50)
                print(f"Strategy:           {strategy_mode}")
                print(f"Overall Sentiment:  {verdict['Sentiment_Score']}/10")
                
                # Only show categories with sufficient data
                valid_categories = verdict.get('Valid_Categories', [])
                if 'Primary' in valid_categories:
                    print(f"� Primary Sources:  {verdict['Primary_Score']}/10")
                if 'Institutional' in valid_categories:
                    print(f"🏛️  Institutional:    {verdict['Institutional_Score']}/10")
                if 'Aggregator' in valid_categories:
                    print(f"📈 Aggregators:      {verdict['Aggregator_Score']}/10")
                if 'Entertainment' in valid_categories:
                    print(f"📺 Entertainment:    {verdict['Entertainment_Score']}/10")
                
                print(f"Price Value:        {verdict['Value_Score']}/10")
                print("-" * 50)
                print(f"FINAL BUY SCORE:  {verdict['Final_Score']}/10")
                print(f"\nREASONING:\n{verdict['Explanation']}")
                print("-" * 50)
                print(f"TOP POSITIVE NEWS:\n>> {best_news}")
                print(f"\nTOP NEGATIVE NEWS:\n>> {worst_news}")
                print("="*50 + "\n")
            
            visualizer.plot_graph(ticker, final_df, profile_data, lookback_days, strategy_mode)
    else:
        print("Scraping failed or no data found.")

def main():
    print("--------------------------------------------------")
    print("   SIGMASTOCKS: AI SENTIMENT & VALUE ANALYZER")
    print("--------------------------------------------------")
    print("Enter 'EXIT' to stop the program")
    print("--------------------------------------------------")
    
    while True:
        ticker = input("\nEnter ticker (e.g., TSLA): ").upper().strip()
        
        if ticker == 'EXIT':
            print("Thank you for using SigmaStocks! Goodbye!")
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
        print("6. MAX (Max Available Data)")
        
        choice = input("Enter choice (e.g., 5D, 2): ").upper().strip()

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
            start_of_year = datetime(today.year, 1, 1)
            delta = today - start_of_year
            lookback_days = delta.days
        elif choice in ['6', 'MAX', '1Y']:
            lookback_days = 365 
        else:
            print("Invalid selection. Defaulting to 1 Month.")
            lookback_days = 30

        print("\nSelect Investment Strategy:")
        print("1. VALUE    (Buy dips - contrarian approach)")
        print("2. MOMENTUM (Follow trends - derivative-based)")
        print("3. BALANCED (Combine both approaches)")
        
        strategy_choice = input("Enter strategy choice (1-3): ").strip()
        
        if strategy_choice == '1':
            strategy_mode = "VALUE"
        elif strategy_choice == '2':
            strategy_mode = "MOMENTUM"
        elif strategy_choice == '3':
            strategy_mode = "BALANCED"
        else:
            print("Invalid selection. Defaulting to BALANCED.")
            strategy_mode = "BALANCED"

        analyze_stock(ticker, lookback_days, strategy_mode)

if __name__ == "__main__":
    main()