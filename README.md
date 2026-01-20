# SigmaStocks: AI Sentiment & Technical Analysis Platform

A sophisticated stock analysis tool that combines advanced technical analysis with hybrid AI-powered sentiment analysis to provide comprehensive investment insights. The system scrapes real-time news data, applies cutting-edge machine learning sentiment analysis, and executes proven technical trading strategies to generate actionable buy/sell scores.

## 🚀 Features

### Dual Trading Strategies
- **VALUE Strategy**: Z-Score based analysis for identifying statistically oversold opportunities
- **MOMENTUM Strategy**: MACD/RSI combination with Golden Cross/Death Cross detection for swing trading and trend following

### Revolutionary Hybrid Sentiment Analysis
- **Hybrid AI System**: Combines VADER's nuanced scoring (-1 to +1) with DistilRoBERTa's financial accuracy
- **Enhanced Financial Lexicon**: 40+ custom financial terms (crash: -3.0, soar: +3.0, beat: +2.5, etc.)
- **Smart Disagreement Resolution**: When models disagree, uses confidence-weighted correction
- **Source Reliability Weighting**: Four-tier categorization system (Primary, Institutional, Aggregators, Entertainment)
- **Real-Time News**: Alpaca News API integration with intelligent filtering

### Advanced Technical Analysis Algorithms
- **Z-Score Analysis**: 50-day rolling window for statistical price deviation
- **MACD**: Moving Average Convergence Divergence with bullish crossover detection
- **RSI**: Relative Strength Index for momentum assessment
- **Golden Cross/Death Cross**: SMA(50) vs SMA(200) major trend signals
- **Buy/Sell Signal Visualization**: Green/red triangles on price charts
- **Falling Knife Protection**: Automatic penalty for stocks with very negative sentiment

### Interactive Visualization
- **Real-Time Charts**: Combined sentiment and price data visualization with trading signals
- **Signal Markers**: Visual buy/sell indicators on matplotlib charts
- **Source Breakdown**: Article count by reliability tier
- **Multiple Timeframes**: 1D, 5D, 1M, 6M analysis periods with strategy-specific restrictions
- **Performance Optimized**: GPU acceleration support for sentiment analysis

## 📊 Technical Specifications

### Hybrid Sentiment Analysis System
The revolutionary hybrid approach combines two complementary AI models:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**:
   - Provides nuanced scoring from -1.0 to +1.0
   - Enhanced with custom financial lexicon
   - Fast processing for real-time analysis

2. **DistilRoBERTa Financial Model**:
   - Fine-tuned for financial news classification
   - High accuracy for positive/negative/neutral detection
   - Prevents major sentiment misclassifications

3. **Hybrid Logic**:
   ```
   If models agree: Use VADER's nuanced score
   If models disagree: Apply confidence-weighted correction
   Formula: abs(vader_score) × ai_confidence × 0.7
   ```

### Financial Algorithms Used
1. **Z-Score**: `Z = (Price - μ) / σ` (50-day window)
2. **MACD**: 12-day EMA - 26-day EMA with 9-day signal line
3. **RSI**: 14-day momentum oscillator
4. **Golden Cross**: SMA(50) > SMA(200) (+2.5 momentum points)
5. **Death Cross**: SMA(50) < SMA(200) (-2.5 momentum points)
6. **Hybrid Sentiment**: VADER + DistilRoBERTa with financial lexicon
7. **Weighted Scoring**: 60% technical + 40% sentiment with penalty system

### Source Reliability Tiers
- **PRIMARY** (Weight: 2.0): SEC filings, Company IR
- **INSTITUTIONAL** (Weight: 1.6-1.8): Bloomberg, WSJ, Reuters
- **AGGREGATORS** (Weight: 1.0-1.5): Yahoo Finance, MarketWatch, Benzinga
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

4. **Set up API keys** (create `.env` file)
   ```bash
   ALPACA_API_KEY=your_alpaca_key
   ALPACA_SECRET_KEY=your_alpaca_secret
   ```

5. **Run the application**
   ```bash
   python src/main.py
   ```

## 🎯 Usage

### Basic Analysis
1. Run the program: `python src/main.py`
2. Enter a stock ticker (e.g., TSLA, AAPL, MSFT)
3. **Select strategy first** (VALUE or MOMENTUM)
4. Choose timeframe based on strategy restrictions
5. View results and interactive charts with trading signals

### Strategy-Specific Timeframes
- **VALUE Strategy**: 1M, 6M, YTD, MAX (long-term analysis)
- **MOMENTUM Strategy**: 1D, 5D, 1M, 6M (short to medium-term signals)

### Output Interpretation
- **Market Sentiment**: 0-10 scale based on hybrid AI sentiment analysis
- **Market Analysis**: 0-10 based on chosen strategy algorithms
- **Final Buy Score**: Combined score with falling knife protection
- **Trading Signals**: Visual buy/sell indicators on charts
- **Top Headlines**: Best/worst sentiment-scored news with confidence scores

## 📈 Strategy Details

### VALUE Strategy
- **Objective**: Identify statistically oversold stocks for long-term holds
- **Signal**: Z-Score < -2.0 with positive sentiment
- **Timeframes**: 1M, 6M, YTD, MAX
- **Best For**: Long-term value investing, contrarian plays
- **Risk**: Catching falling knives in declining markets

### MOMENTUM Strategy  
- **Objective**: Capture trend continuation, breakouts, and major trend changes
- **Signals**: 
  - MACD bullish crossover + RSI not overbought
  - Golden Cross (SMA50 > SMA200) = major bullish signal
  - Death Cross (SMA50 < SMA200) = major bearish signal
- **Timeframes**: 1D, 5D, 1M, 6M
- **Best For**: Swing trading, trend following, momentum plays
- **Risk**: Whipsaws in sideways markets

## 🤖 AI Training System

### Autonomous Learning
- **Location**: `training/autonomous_trainer.py`
- **Capability**: Self-improving sentiment analysis through objective performance metrics
- **Features**: 
  - Automatic sentiment error detection
  - Model weight adjustments without human intervention
  - Performance tracking across multiple tickers and strategies

## ⚠️ Disclaimer

**This tool is for educational and research purposes only. Do not rely solely on SigmaStocks for investment decisions. Always conduct your own research and consider consulting with financial professionals before making investment choices.**

## 🔧 Requirements

- Python 3.8+
- Internet connection for real-time data
- Optional: CUDA-compatible GPU for faster sentiment analysis
- See `requirements.txt` for complete dependency list

## 📦 Dependencies

### Core Analysis
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `yfinance>=0.2.0` - Financial data retrieval

### Hybrid Sentiment Analysis
- `vaderSentiment>=3.3.2` - Nuanced sentiment scoring
- `transformers>=4.30.0` - DistilRoBERTa model
- `torch>=2.0.0` - PyTorch for neural networks
- `tokenizers>=0.13.0` - Text tokenization

### News & Visualization
- `alpaca-py>=0.7.0` - Professional news data
- `matplotlib>=3.7.0` - Chart visualization
- `tqdm>=4.65.0` - Progress bars

## 📝 License

This project is licensed under the GNU GPLv3 License - see the LICENSE file for details.

---

*Built with Python, Hybrid AI (VADER + DistilRoBERTa), yfinance, and matplotlib*
