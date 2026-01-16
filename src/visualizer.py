import matplotlib.pyplot as plt

def plot_graph(ticker, merged_df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    colors = ['green' if x > 0 else 'red' for x in merged_df['Buy_Score']]
    ax1.bar(merged_df.index, merged_df['Buy_Score'], color=colors, alpha=0.6, label='Buy/Sell Signal')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Algorithm Score (Sentiment - Momentum)', color='green')
    ax1.axhline(0, color='grey', linewidth=0.8, linestyle='--')

    ax2 = ax1.twinx()
    ax2.plot(merged_df.index, merged_df['Close'], color='black', linewidth=2, linestyle='-', label='Stock Price')
    ax2.set_ylabel('Stock Price ($)', color='black')

    plt.title(f'{ticker} Algo Strategy: "Buy the Dip"')
    plt.show()