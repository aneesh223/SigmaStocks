import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_graph(ticker, merged_df, company_info, timeframe_days=30, strategy_mode="BALANCED"):
    name, industry, mkt_cap = company_info
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    score_column = 'Compound_Score' if 'Compound_Score' in merged_df.columns else 'Buy_Score'
    colors = ['green' if x > 0 else 'red' for x in merged_df[score_column]]
    
    if timeframe_days <= 1:
        bar_width = 0.02
    else:
        bar_width = 0.8
    
    ax1.bar(merged_df.index, merged_df[score_column], color=colors, alpha=0.6, 
            label='Sentiment Score', width=bar_width)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Sentiment Score', color='green')
    ax1.axhline(0, color='grey', linewidth=0.8, linestyle='--')

    if timeframe_days <= 1:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, timeframe_days//10)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    if 'Close' in merged_df.columns and not merged_df['Close'].isna().all():
        ax2 = ax1.twinx()
        price_data = merged_df['Close'].dropna()
        if not price_data.empty:
            if timeframe_days <= 1:
                ax2.plot(price_data.index, price_data.values, color='blue', linewidth=3, 
                        linestyle='-', label='Stock Price', marker='o', markersize=4, alpha=0.8)
            else:
                ax2.plot(price_data.index, price_data.values, color='black', linewidth=2, 
                        linestyle='-', label='Stock Price', marker='o', markersize=3)
            
            ax2.set_ylabel('Stock Price ($)', color='blue' if timeframe_days <= 1 else 'black')
            ax2.tick_params(axis='y', labelcolor='blue' if timeframe_days <= 1 else 'black')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            print(f"Price line plotted with {len(price_data)} data points")
        else:
            print("No valid price data to plot")
    else:
        print("No Close price column found or all values are NaN")

    if timeframe_days <= 1:
        title_suffix = f"Hourly Sentiment Analysis ({strategy_mode} Strategy)"
    else:
        title_suffix = f"Daily Sentiment Analysis ({strategy_mode} Strategy)"
        
    plt.title(f'{name} ({ticker}) - {title_suffix}')
    
    info_text = f"Industry: {industry}   |   Market Cap: {mkt_cap}"
    
    plt.figtext(0.5, 0.02, info_text, ha="center", fontsize=10, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.subplots_adjust(bottom=0.15)
    
    plt.tight_layout()
    plt.show()