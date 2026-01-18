import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better macOS compatibility
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime

def plot_graph(ticker, merged_df, company_info, timeframe_days=30, strategy_mode="BALANCED", save_plot=False):
    name, industry, mkt_cap = company_info
    
    # Create figure with clean styling
    plt.style.use('default')
    fig = plt.figure(figsize=(14, 10))
    
    # Check if we have enough data for article breakdown
    has_article_data = all(col in merged_df.columns for col in ['Primary_Count', 'Institutional_Count', 'Aggregator_Count', 'Entertainment_Count'])
    MIN_ARTICLES_FOR_DISPLAY = 3
    
    if has_article_data:
        # Check if any category has sufficient articles
        article_totals = {
            'Primary': merged_df['Primary_Count'].sum(),
            'Institutional': merged_df['Institutional_Count'].sum(), 
            'Aggregator': merged_df['Aggregator_Count'].sum(),
            'Entertainment': merged_df['Entertainment_Count'].sum()
        }
        show_breakdown = any(total >= MIN_ARTICLES_FOR_DISPLAY for total in article_totals.values())
    else:
        show_breakdown = False
    
    # Create subplots based on available data
    if show_breakdown:
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5], hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax3 = fig.add_subplot(gs[1])
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 0.5], hspace=0.2)
        ax1 = fig.add_subplot(gs[0])

    # Main sentiment plot with clean line chart
    score_column = 'Compound_Score' if 'Compound_Score' in merged_df.columns else 'Buy_Score'
    
    # Define bar_width for article breakdown subplot
    bar_width = 0.02 if timeframe_days <= 1 else 0.8
    
    # Create sentiment line with color based on sentiment
    sentiment_data = merged_df[score_column].dropna()
    if not sentiment_data.empty:
        # Color the line based on overall sentiment trend
        avg_sentiment = sentiment_data.mean()
        if avg_sentiment > 0.1:
            line_color = '#2E8B57'  # Green for positive
        elif avg_sentiment < -0.1:
            line_color = '#DC143C'  # Red for negative
        else:
            line_color = '#708090'  # Gray for neutral
        
        ax1.plot(sentiment_data.index, sentiment_data.values, color=line_color, linewidth=2.5, 
                label='Overall Sentiment', marker='o', markersize=4, alpha=0.8)

    # Style the main plot
    ax1.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
    ax1.axhline(0, color='#696969', linewidth=1, linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_facecolor('#FAFAFA')
    
    # Add legend for sentiment line
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Format x-axis
    if timeframe_days <= 1:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    else:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, timeframe_days//10)))
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, fontsize=10)

    # Price plot on secondary axis
    if 'Close' in merged_df.columns and not merged_df['Close'].isna().all():
        ax2 = ax1.twinx()
        price_data = merged_df['Close'].dropna()
        if not price_data.empty:
            ax2.plot(price_data.index, price_data.values, color='#800080', linewidth=2.5, 
                    linestyle='-', alpha=0.8, zorder=10)
            
            ax2.set_ylabel('Stock Price ($)', color='#800080', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#800080', labelsize=10)
            ax2.grid(False)  # Don't duplicate grid

    # Article count subplot (only if we have breakdown data)
    if show_breakdown:
        primary_counts = merged_df['Primary_Count'].fillna(0)
        institutional_counts = merged_df['Institutional_Count'].fillna(0)
        aggregator_counts = merged_df['Aggregator_Count'].fillna(0)
        entertainment_counts = merged_df['Entertainment_Count'].fillna(0)
        
        # Stack bars only for categories with sufficient data
        bottom = pd.Series(0, index=merged_df.index)
        
        if article_totals['Primary'] >= MIN_ARTICLES_FOR_DISPLAY:
            ax3.bar(merged_df.index, primary_counts, color='#FFD700', alpha=0.8, 
                    label=f"Primary ({int(article_totals['Primary'])})", width=bar_width)
            bottom += primary_counts
        
        if article_totals['Institutional'] >= MIN_ARTICLES_FOR_DISPLAY:
            ax3.bar(merged_df.index, institutional_counts, bottom=bottom, 
                    color='#4169E1', alpha=0.7, label=f"Institutional ({int(article_totals['Institutional'])})", width=bar_width)
            bottom += institutional_counts
        
        if article_totals['Aggregator'] >= MIN_ARTICLES_FOR_DISPLAY:
            ax3.bar(merged_df.index, aggregator_counts, bottom=bottom, 
                    color='#32CD32', alpha=0.6, label=f"Aggregators ({int(article_totals['Aggregator'])})", width=bar_width)
            bottom += aggregator_counts
        
        if article_totals['Entertainment'] >= MIN_ARTICLES_FOR_DISPLAY:
            ax3.bar(merged_df.index, entertainment_counts, bottom=bottom,
                    color='#FF6347', alpha=0.5, label=f"Entertainment ({int(article_totals['Entertainment'])})", width=bar_width)
        
        ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Articles', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('#FAFAFA')
        
        # Format x-axis for article count plot
        if timeframe_days <= 1:
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        else:
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, timeframe_days//10)))
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, fontsize=10)

    # Clean title and info
    title_suffix = f"Sentiment Analysis - {strategy_mode} Strategy"
    ax1.set_title(f'{name} ({ticker}) - {title_suffix}', fontsize=16, fontweight='bold', pad=20)
    
    # Company info at bottom with better styling
    info_text = f"Industry: {industry}  •  Market Cap: {mkt_cap}"
    fig.text(0.5, 0.02, info_text, ha="center", fontsize=11, 
             bbox={"facecolor":"#E6E6FA", "alpha":0.8, "pad":8, "boxstyle":"round,pad=0.5"})
    
    # Adjust layout with manual spacing to avoid tight_layout warnings
    if show_breakdown:
        # Three subplot layout
        plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.12, hspace=0.4)
    else:
        # Two subplot layout  
        plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.12, hspace=0.3)
    
    if save_plot:
        # Save plot instead of showing it
        filename = f"{ticker}_{strategy_mode}_{timeframe_days}d_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
        plt.close()  # Close the figure to free memory
    else:
        # Use non-blocking show to prevent IDE hanging
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to ensure plot renders