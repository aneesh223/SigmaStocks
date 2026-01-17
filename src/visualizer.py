import matplotlib.pyplot as plt

def plot_graph(ticker, merged_df, company_info):
    """
    company_info is a tuple: (Name, Industry, Market Cap)
    """
    name, industry, mkt_cap = company_info
    
    fig, ax1 = plt.subplots(figsize=(12, 7)) # Increased height for info text

    # Bar Chart
    colors = ['green' if x > 0 else 'red' for x in merged_df['Buy_Score']]
    ax1.bar(merged_df.index, merged_df['Buy_Score'], color=colors, alpha=0.6, label='Buy/Sell Signal')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Algo Score', color='green')
    ax1.axhline(0, color='grey', linewidth=0.8, linestyle='--')

    # Line Chart
    ax2 = ax1.twinx()
    ax2.plot(merged_df.index, merged_df['Close'], color='black', linewidth=2, linestyle='-', label='Stock Price')
    ax2.set_ylabel('Stock Price ($)', color='black')

    # Title
    plt.title(f'{name} ({ticker}) - Algo Strategy')
    
    # --- NEW: ADD INFO BOX BELOW CHART ---
    # We use figtext (figure coordinates: 0,0 is bottom-left, 1,1 is top-right)
    info_text = f"Industry: {industry}   |   Market Cap: {mkt_cap}"
    
    plt.figtext(0.5, 0.02, info_text, ha="center", fontsize=10, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Adjust layout to make room for the text at the bottom
    plt.subplots_adjust(bottom=0.15)
    # -------------------------------------
    
    plt.show()