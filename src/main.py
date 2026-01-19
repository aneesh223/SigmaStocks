# Memory-optimized imports
from datetime import datetime, timedelta
import pytz
import scraper
import analyzer
import market
import visualizer

def analyze_stock(ticker, lookback_days, strategy_mode, testing_mode=False, custom_date=None):
    """Optimized stock analysis with memory efficiency"""
    if testing_mode and custom_date:
        print(f"\n🧪 Testing mode: {custom_date.strftime('%B %d, %Y')}")
    
    print(f"\nAnalyzing {ticker}...")
    
    try:
        # Memory-efficient data fetching
        profile_data, raw_data = scraper.scrape_hybrid(ticker, days=lookback_days)
        company_name, industry, mkt_cap = profile_data
        
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if not raw_data:
        print("No data found. Check your connection.")
        return
    
    # Optimized sentiment analysis
    sentiment_df = analyzer.get_sentiment(raw_data, ticker, lookback_days)
    best_news, worst_news = analyzer.get_top_headlines_wrapper(raw_data, ticker, lookback_days)
    
    # Optimized date filtering
    today = custom_date if testing_mode and custom_date else datetime.now()
        
    if lookback_days <= 1:
        cutoff_date = today.date()
        sentiment_df = sentiment_df[sentiment_df.index.date == cutoff_date]
    else:
        cutoff_date = (today - timedelta(days=lookback_days)).date()
        if hasattr(sentiment_df.index[0], 'date'):
            sentiment_df.index = sentiment_df.index.date
        sentiment_df = sentiment_df[sentiment_df.index >= cutoff_date]

    if sentiment_df.empty:
        print(f"No relevant news found for {ticker}.")
        return
    
    # Calculate verdict and display results
    verdict = market.calculate_verdict(ticker, sentiment_df, strategy_mode.lower(), lookback_days, custom_date)
    
    if verdict:
        print("\n" + "="*50)
        print(f"   ANALYSIS FOR {company_name} ({ticker})")
        print("="*50)
        print(f"Industry:   {industry}")
        print(f"Market Cap: {mkt_cap}")
        print("-" * 50)
        print(f"Strategy:           {verdict['Strategy']}")
        print(f"Market Sentiment:   {verdict['Sentiment_Health']}/10")
        print(f"Market Analysis:    {verdict['Technical_Score']}/10")
        print("-" * 50)
        print(f"FINAL BUY SCORE:    {verdict['Final_Buy_Score']}/10")
        print(f"\nREASONING:\n{verdict['Explanation']}")
        print("-" * 50)
        print(f"TOP POSITIVE NEWS:\n>> {best_news}")
        print(f"\nTOP NEGATIVE NEWS:\n>> {worst_news}")
        print("="*50 + "\n")
    
    # Generate optimized visualization
    final_df = market.get_visualization_data(ticker, sentiment_df, lookback_days)
    visualizer.plot_graph(ticker, final_df, profile_data, lookback_days, strategy_mode)

def main():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   📈 SIGMASTOCKS: AI SENTIMENT ANALYZER")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    testing_mode = False
    custom_date = None
    
    while True:
        ticker = input("\n🎯 Enter ticker (or 'exit'): ").upper().strip()
        
        if ticker in ['EXIT', 'QUIT', 'Q']: 
            break
        
        if ticker == 'DEBUG':
            print("\n🧪 Debug mode activated")
            try:
                date_input = input("Enter date (YYYY-MM-DD): ").strip()
                custom_date = datetime.strptime(date_input, "%Y-%m-%d")
                testing_mode = True
                print(f"✅ Debug date set: {custom_date.strftime('%Y-%m-%d')}")
            except ValueError:
                print("❌ Invalid date format")
            continue

        if not ticker: 
            continue

        # Keep current timeframe options (as requested)
        print("\nSelect Timeframe:")
        print("1. 1D  (1 Day)")
        print("2. 5D  (5 Days)")
        print("3. 1M  (1 Month)")
        print("4. 6M  (6 Months)")
        print("5. YTD (Year-To-Date)")
        print("6. 1Y  (1 Year)")
        
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
        elif choice in ['6', '1Y']:
            lookback_days = 365 
        else:
            print("Invalid selection. Defaulting to 1 Month.")
            lookback_days = 30

        print("\nSelect Investment Strategy:")
        print("1. VALUE    (Buy the Dip)")
        print("2. MOMENTUM (Swing Trading)")
        
        strategy_choice = input("Enter strategy choice (1-2): ").strip()
        
        if strategy_choice == '1':
            strategy_mode = "VALUE"
        elif strategy_choice == '2':
            strategy_mode = "MOMENTUM"
        else:
            print("Invalid selection. Defaulting to VALUE.")
            strategy_mode = "VALUE"

        analyze_stock(ticker, lookback_days, strategy_mode, testing_mode, custom_date)
        
        if testing_mode:
            testing_mode = False
            custom_date = None

if __name__ == "__main__":
    main()