# SigmaStocks: AI Sentiment & Technical Analysis Platform

A sophisticated stock analysis tool that combines advanced technical analysis with AI-powered sentiment analysis to provide comprehensive investment insights. The system scrapes real-time news data, applies machine learning sentiment analysis, and executes proven technical trading strategies to generate actionable buy/sell scores.

## 🚀 Features

### Dual Trading Strategies
- **VALUE Strategy**: Z-Score based analysis for identifying statistically oversold opportunities
- **MOMENTUM Strategy**: MACD/RSI combination for swing trading and trend following

### Advanced Sentiment Analysis
- **AI-Powered**: VADER sentiment analysis enhanced with financial lexicon
- **Source Reliability Weighting**: Four-tier categorization system (Primary, Institutional, Aggregators, Entertainment)
- **Real-Time News**: Google News RSS integration with intelligent filtering

### Technical Analysis Algorithms
- **Z-Score Analysis**: 50-day rolling window for statistical price deviation
- **MACD**: Moving Average Convergence Divergence with bullish crossover detection
- **RSI**: Relative Strength Index for momentum assessment
- **Falling Knife Protection**: Automatic penalty for stocks with very negative sentiment

### Interactive Visualization
- **Real-Time Charts**: Combined sentiment and price data visualization
- **Source Breakdown**: Article count by reliability tier
- **Multiple Timeframes**: 1D, 5D, 1M, 6M, YTD, MAX analysis periods
- **Debug Mode**: Historical analysis with custom date simulation

## 📊 Technical Specifications

### Financial Algorithms Used
1. **Z-Score**: `Z = (Price - μ) / σ` (50-day window)
2. **MACD**: 12-day EMA - 26-day EMA with 9-day signal line
3. **RSI**: 14-day momentum oscillator
4. **VADER Sentiment**: Enhanced with financial terminology
5. **Weighted Scoring**: 60% technical + 40% sentiment with penalty system

### Source Reliability Tiers
- **PRIMARY** (Weight: 2.0): SEC filings, Company IR
- **INSTITUTIONAL** (Weight: 1.6-1.8): Bloomberg, WSJ, Reuters
- **AGGREGATORS** (Weight: 1.0-1.2): Yahoo Finance, MarketWatch
- **ENTERTAINMENT** (Weight: 0.3-0.8): Motley Fool, Reddit, Social Media

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sigmastocks.git
   cd sigmastocks
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python src/main.py
   ```

## 🎯 Usage

### Basic Analysis
1. Run the program: `python src/main.py`
2. Enter a stock ticker (e.g., TSLA, AAPL, MSFT)
3. Select timeframe (1D, 5D, 1M, 6M, YTD, MAX)
4. Choose strategy (VALUE or MOMENTUM)
5. View results and interactive charts

### Debug Mode
- Type `DEBUG` as ticker to activate testing mode
- Enter custom date (YYYY-MM-DD format)
- Analyze historical performance with simulated dates

### Output Interpretation
- **Sentiment Health**: 0-10 scale based on news sentiment
- **Technical Score**: 0-10 based on chosen strategy algorithms
- **Final Buy Score**: Combined score with falling knife protection
- **Reasoning**: Detailed explanation of analysis results

## 📈 Strategy Details

### VALUE Strategy
- **Objective**: Identify statistically oversold stocks
- **Signal**: Z-Score < -2.0 with positive sentiment
- **Best For**: Long-term value investing, contrarian plays
- **Risk**: Catching falling knives in declining markets

### MOMENTUM Strategy  
- **Objective**: Capture trend continuation and breakouts
- **Signal**: MACD bullish crossover + RSI not overbought
- **Best For**: Swing trading, trend following
- **Risk**: Whipsaws in sideways markets

## ⚠️ Disclaimer

**This tool is for educational and research purposes only. Do not rely solely on SigmaStocks for investment decisions. Always conduct your own research and consider consulting with financial professionals before making investment choices.**

## 🔧 Requirements

- Python 3.8+
- Internet connection for real-time data
- See `requirements.txt` for complete dependency list

## 📝 License

This project is licensed under the GNU GPLv3 License - see the LICENSE file for details.

---

*Built with Python, yfinance, NLTK, and matplotlib*
