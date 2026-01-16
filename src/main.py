from datetime import datetime, timedelta
import scraper
import analyzer
import market
import visualizer

def main():
    ticker = input("Enter ticker (e.g., TSLA): ")

    print("\nSelect Time Range:")
    print("1. Past 1 Week")
    print("2. Past 2 Weeks")
    print("3. Past 1 Month")
    print("4. All Available Data")
    choice = input("Enter number (1-4): ")

    raw_data = scraper.scrape_finviz(ticker)
    
    if raw_data:
        sentiment_df = analyzer.get_sentiment(raw_data)
        
        today = datetime.now().date()
        cutoff_date = None
        if choice == '1':
            cutoff_date = today - timedelta(days=7)
        elif choice == '2':
            cutoff_date = today - timedelta(days=14)
        elif choice == '3':
            cutoff_date = today - timedelta(days=30)
            
        if cutoff_date:
            sentiment_df = sentiment_df[sentiment_df.index >= cutoff_date]

        if sentiment_df.empty:
            print("No news found in this date range.")
        else:
            final_df = market.get_financials(ticker, sentiment_df)
            
            visualizer.plot_graph(ticker, final_df)
    else:
        print("Scraping failed.")

if __name__ == "__main__":
    main()