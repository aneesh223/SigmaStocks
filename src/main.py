from datetime import datetime, timedelta
import pytz
import scraper
import analyzer
import market
import visualizer

def analyze_stock(ticker, lookback_days, strategy_mode, testing_mode=False, custom_date=None):
    if testing_mode and custom_date:
        print(f"\n🧪 DEBUG MODE ANALYSIS")
        print(f"📅 Analysis Date: {custom_date.strftime('%B %d, %Y')}")
        print(f"📊 Looking back {lookback_days} days from debug date")
        print(f"🎯 Strategy: {strategy_mode}")
    
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
        best_news, worst_news = analyzer.get_top_headlines_wrapper(raw_data, ticker, lookback_days)
        
        # Date handling with testing mode support
        if testing_mode and custom_date:
            today = custom_date
            print(f"🧪 TESTING MODE: Using {today.strftime('%B %d, %Y')} as analysis date")
        else:
            today = datetime.now()
            
        # For 1-day analysis, we want data only for that specific day
        if lookback_days <= 1:
            cutoff_date = today.date()
            sentiment_df = sentiment_df[sentiment_df.index.date == cutoff_date]
        else:
            # For multi-day analysis, calculate normal cutoff
            cutoff_date = (today - timedelta(days=lookback_days)).date()
            if hasattr(sentiment_df.index[0], 'date'):
                sentiment_df.index = sentiment_df.index.date
            sentiment_df = sentiment_df[sentiment_df.index >= cutoff_date]

        if sentiment_df.empty:
            print(f"No relevant news found for {ticker} in the last {lookback_days} days.")
        else:
            verdict = market.calculate_verdict(ticker, sentiment_df, strategy_mode.lower(), lookback_days, custom_date)
            
            if verdict:
                print("\n" + "="*50)
                print(f"   ANALYSIS FOR {company_name} ({ticker})")
                print("="*50)
                print(f"Industry:   {industry}")
                print(f"Market Cap: {mkt_cap}")
                print("-" * 50)
                print(f"Strategy:           {verdict['Strategy']}")
                print(f"Sentiment Health:   {verdict['Sentiment_Health']}/10")
                print(f"Technical Score:    {verdict['Technical_Score']}/10")
                print("-" * 50)
                print(f"FINAL BUY SCORE:    {verdict['Final_Buy_Score']}/10")
                print(f"\nREASONING:\n{verdict['Explanation']}")
                print("-" * 50)
                print(f"TOP POSITIVE NEWS:\n>> {best_news}")
                print(f"\nTOP NEGATIVE NEWS:\n>> {worst_news}")
                print("="*50 + "\n")
            
            # Get data for visualization
            final_df = market.get_visualization_data(ticker, sentiment_df, lookback_days)
            visualizer.plot_graph(ticker, final_df, profile_data, lookback_days, strategy_mode)
    else:
        print("Scraping failed or no data found.")

def main():
    print("--------------------------------------------------")
    print("   SIGMASTOCKS: AI SENTIMENT & VALUE ANALYZER")
    print("--------------------------------------------------")
    print("Enter 'EXIT' to stop the program")
    print("Enter 'DEBUG' to activate testing mode")
    print("--------------------------------------------------")
    
    testing_mode = False
    custom_date = None
    
    while True:
        ticker = input("\nEnter ticker (e.g., TSLA): ").upper().strip()
        
        if ticker == 'EXIT':
            print("Thank you for using SigmaStocks! Goodbye!")
            break
        
        if ticker == 'DEBUG':
            print("\n🧪 --- DEBUG MODE ACTIVATED ---")
            print("Enter a custom date to simulate analysis from that point in time.")
            print("Format: YYYY-MM-DD (e.g., 2026-01-15)")
            
            while True:
                date_input = input("Enter test date: ").strip()
                try:
                    # Parse the date input
                    year, month, day = map(int, date_input.split('-'))
                    custom_date = datetime(year, month, day)
                    print(f"✅ Debug mode set to: {custom_date.strftime('%B %d, %Y')}")
                    testing_mode = True
                    break
                except (ValueError, TypeError):
                    print("❌ Invalid date format. Please use YYYY-MM-DD (e.g., 2026-01-15)")
                    continue
            
            # Now get the actual ticker
            ticker = input("\nEnter ticker for debug analysis (e.g., TSLA): ").upper().strip()
            
            if not ticker or ticker in ['EXIT', 'DEBUG']:
                print("Please enter a valid ticker symbol.")
                continue

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

        # Use testing date if in testing mode, otherwise current date
        if testing_mode and custom_date:
            today = custom_date
        else:
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
        print("1. VALUE    (Buy the Dip - Z-Score based)")
        print("2. MOMENTUM (Swing Trading - MACD/RSI based)")
        
        strategy_choice = input("Enter strategy choice (1-2): ").strip()
        
        if strategy_choice == '1':
            strategy_mode = "VALUE"
        elif strategy_choice == '2':
            strategy_mode = "MOMENTUM"
        else:
            print("Invalid selection. Defaulting to VALUE.")
            strategy_mode = "VALUE"

        analyze_stock(ticker, lookback_days, strategy_mode, testing_mode, custom_date)
        
        # Reset debug mode after each analysis
        if testing_mode:
            print(f"\n🧪 Debug analysis complete. Returning to normal mode.")
            testing_mode = False
            custom_date = None

if __name__ == "__main__":
    main()