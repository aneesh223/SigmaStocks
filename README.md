# SigmaStocks: AI Sentiment & Technical Analysis Platform

A sophisticated stock analysis tool that combines advanced technical analysis with hybrid AI-powered sentiment analysis to provide comprehensive investment insights. The system scrapes real-time news data, applies cutting-edge machine learning sentiment analysis, and executes proven technical trading strategies to generate actionable buy/sell scores.

## 🚀 Features

### Dual Trading Strategies
- **VALUE Strategy**: Rolling window Final_Buy_Score analysis for identifying statistically oversold opportunities
- **MOMENTUM Strategy**: MACD/RSI combination with Golden Cross/Death Cross detection for swing trading and trend following

### Revolutionary Hybrid Sentiment Analysis
- **Hybrid AI System**: Combines VADER's nuanced scoring (-1 to +1) with DistilRoBERTa's financial accuracy
- **Enhanced Financial Lexicon**: 40+ custom financial terms (crash: -3.0, soar: +3.0, beat: +2.5, etc.)
- **Smart Disagreement Resolution**: When models disagree, uses confidence-weighted correction
- **Source Reliability Weighting**: Four-tier categorization system (Primary, Institutional, Aggregators, Entertainment)
- **Real-Time News**: Alpaca News API integration with intelligent filtering

### Advanced Technical Analysis Algorithms
- **Rolling Window Analysis**: 3-5 day rolling window with aggressive recency weighting (decay_rate=0.5)
- **B(t) Calculation**: Individual buy scores at each time period with exponential decay
- **Final_Buy_Score**: Weighted average of recent B(t) values for stable signals
- **MACD**: Moving Average Convergence Divergence with bullish crossover detection
- **RSI**: Relative Strength Index for momentum assessment
- **Golden Cross/Death Cross**: SMA(50) vs SMA(200) major trend signals
- **Buy/Sell Signal Visualization**: Green/red triangles on price charts
- **Adaptive Risk Management**: Dynamic thresholds, stop-loss, take-profit, and trailing stops

### Interactive Visualization
- **Real-Time Charts**: Combined sentiment and price data visualization with trading signals
- **Signal Markers**: Visual buy/sell indicators on matplotlib charts
- **Source Breakdown**: Article count by reliability tier
- **Multiple Timeframes**: 1D, 5D, 1M, 6M analysis periods with strategy-specific restrictions
- **Performance Optimized**: Memory-efficient processing with intelligent caching

### Comprehensive Backtesting System
- **Historical Analysis**: Test strategies against years of Alpaca market data
- **Realistic Simulation**: Day-by-day execution with no look-ahead bias
- **Advanced Risk Management**: Stop-loss (-8%), take-profit (+25%), trailing stops (-10%)
- **Market Regime Detection**: Automatic adaptation to BULL/BEAR/SIDEWAYS markets
- **Performance Analytics**: Detailed comparison vs buy-and-hold with alpha calculation
- **CLI Interface**: Streamlined command-line backtesting with batch processing

## 📊 Technical Specifications

### Time-Series Algorithm Architecture
The system uses a sophisticated rolling window approach:

1. **B(t) Calculation**: Individual buy scores at each time period
   ```
   B(t) = Weighted average of sentiment scores at time t
   Recency weighting: exp(-decay_rate × age_in_days)
   ```

2. **Final_Buy_Score Calculation**: Rolling window analysis
   ```
   Final_Buy_Score(t) = Σ(B(i) × weight(i)) for i in [t-window, t]
   Window size: 3-5 days with aggressive decay (0.5)
   ```

3. **Market Regime Adaptation**:
   - **BULL Market**: Wider stop-loss (-12%), higher take-profit (+40%), larger positions (1.2x)
   - **BEAR Market**: Tighter stop-loss (-6%), lower take-profit (+15%), smaller positions (0.8x)
   - **SIDEWAYS**: Standard parameters with dynamic threshold adjustment

### Hybrid Sentiment Analysis System
The revolutionary hybrid approach combines two complementary AI models:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**:
   - Provides nuanced scoring from -1.0 to +1.0
   - Enhanced with custom financial lexicon
   - Fast processing for real-time analysis

2. **DistilRoBERTa Financial Model**:
   - Fine-tuned for financial news classification (mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
   - High accuracy for positive/negative/neutral detection
   - Neutral dampening logic (if Neutral > 0.6, multiply score by 0.2)

3. **Hybrid Logic**:
   ```
   If models agree: Use VADER's nuanced score
   If models disagree: Apply confidence-weighted correction
   Formula: abs(vader_score) × ai_confidence × 0.7
   ```

### Financial Algorithms Used
1. **Rolling Window Final_Buy_Score**: 3-5 day window with exponential decay
2. **MACD**: 12-day EMA - 26-day EMA with 9-day signal line
3. **RSI**: 14-day momentum oscillator
4. **Golden Cross**: SMA(50) > SMA(200) (+2.5 momentum points)
5. **Death Cross**: SMA(50) < SMA(200) (-2.5 momentum points)
6. **Hybrid Sentiment**: VADER + DistilRoBERTa with financial lexicon
7. **Dynamic Risk Management**: Adaptive thresholds based on score volatility

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

## 🎯 Usage

### Real-Time Analysis
1. Run the program: `python src/main.py`
2. Enter a stock ticker (e.g., TSLA, AAPL, MSFT)
3. **Select strategy first** (VALUE or MOMENTUM)
4. Choose timeframe based on strategy restrictions
5. View results and interactive charts with trading signals

### Historical Backtesting
1. **Single Backtest**: `python backtester/run_backtest.py TSLA m 30`
   - Format: `<ticker> <strategy> <period> [cash] [--plot] [--save]`
   - Strategy: `v` (VALUE) or `m` (MOMENTUM)
   - Period: Number of days to backtest

2. **Batch Testing**: `python backtester/batch_backtest.py`
   - Tests multiple configurations automatically
   - Generates performance comparison reports

3. **Examples**: `python backtester/examples.py`
   - Shows usage examples and parameter explanations

### Strategy-Specific Timeframes
- **VALUE Strategy**: 1M, 6M, YTD, MAX (long-term analysis)
- **MOMENTUM Strategy**: 1D, 5D, 1M, 6M (short to medium-term signals)

### Output Interpretation
- **Market Sentiment**: 0-10 scale based on hybrid AI sentiment analysis
- **Market Analysis**: 0-10 based on chosen strategy algorithms
- **Final Buy Score**: Combined score with rolling window analysis
- **Trading Signals**: Visual buy/sell indicators on charts
- **Top Headlines**: Best/worst sentiment-scored news with confidence scores

## 📈 Strategy Details

### VALUE Strategy
- **Objective**: Identify statistically oversold stocks using rolling window analysis
- **Algorithm**: Final_Buy_Score with 5-day rolling window and recency weighting
- **Timeframes**: 1M, 6M, YTD, MAX
- **Best For**: Long-term value investing, contrarian plays
- **Risk Management**: Conservative thresholds with capital preservation focus

### MOMENTUM Strategy  
- **Objective**: Capture trend continuation, breakouts, and major trend changes
- **Signals**: 
  - MACD bullish crossover + RSI not overbought
  - Golden Cross (SMA50 > SMA200) = major bullish signal
  - Death Cross (SMA50 < SMA200) = major bearish signal
- **Timeframes**: 1D, 5D, 1M, 6M
- **Best For**: Swing trading, trend following, momentum plays
- **Risk Management**: Dynamic thresholds with aggressive profit-taking

## 📈 Backtesting System

### Comprehensive Strategy Testing
- **Location**: `backtester/` folder
- **Capability**: Test strategies against historical Alpaca data with realistic simulation
- **Features**:
  - Real Alpaca market data integration (11+ years of data back to 2014)
  - Day-by-day simulation with no look-ahead bias
  - Pre-calculated B(t) and Final_Buy_Score for maximum efficiency
  - Advanced risk management with stop-loss, take-profit, and trailing stops
  - Market regime detection and adaptive parameters
  - Performance analysis vs buy-and-hold with detailed trade logs
  - Visual performance charts and P&L tracking

### CLI Usage Examples
```bash
# Basic backtest: TSLA momentum strategy, 30 days
python backtester/run_backtest.py TSLA m 30

# Advanced backtest: AAPL value strategy, 180 days, $50k capital, save results
python backtester/run_backtest.py AAPL v 180 50000 --save

# Batch testing multiple configurations
python backtester/batch_backtest.py

# View usage examples
python backtester/examples.py
```

### Performance Optimization
- **Efficient Data Processing**: All sentiment analysis done once at startup
- **Pre-calculated Scores**: B(t) and Final_Buy_Score calculated in advance
- **Fast Simulation**: Lookup-based trading decisions for maximum speed
- **Memory Management**: Optimized data structures and garbage collection

## ⚠️ Disclaimer

**This tool is for educational and research purposes only. Do not rely solely on SigmaStocks for investment decisions. Always conduct your own research and consider consulting with financial professionals before making investment choices.**

## 🔧 Requirements

- Python 3.8+
- Internet connection for real-time data
- Alpaca API keys for news and historical data
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

### Backtesting
- `pytz>=2023.3` - Timezone handling for historical data
- `concurrent.futures` - Parallel processing for efficiency

## 📝 License

This project is licensed under the GNU GPLv3 License - see the LICENSE file for details.

---

*Built with Python, Hybrid AI (VADER + DistilRoBERTa), Advanced Risk Management, and Comprehensive Backtesting*
