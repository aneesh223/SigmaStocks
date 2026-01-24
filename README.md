# Orthrus: AI-Powered Algorithmic Trading Engine

**An advanced algorithmic trading platform that combines DistilRoBERTa transformer models with sophisticated technical analysis to generate data-driven investment insights. Features comprehensive backtesting, adaptive risk management, and market regime detection for systematic trading strategies.**

## 🎯 **What Makes Orthrus Special**

- **🤖 Hybrid AI Sentiment Analysis (Hence the Name)**: Combines VADER + DistilRoBERTa for superior financial news interpretation
- **📈 Adaptive Market Regime Detection**: Automatically adjusts strategy based on BULL/BEAR/SIDEWAYS market conditions  
- **🎯 Comprehensive Testing Framework**: Extensive backtesting across multiple market scenarios and timeframes
- **🛡️ Advanced Risk Management**: Dynamic stop-loss, take-profit, and volatility protection systems
- **⚡ Production-Ready**: Professional-grade backtesting system with fair performance comparisons

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
- **Dual Interface**: Interactive mode (`input.py`) and headless CLI mode (`headless.py`)
- **Signal Markers**: Visual buy/sell indicators on matplotlib charts
- **Source Breakdown**: Article count by reliability tier
- **Multiple Timeframes**: 1D, 5D, 1M, 6M analysis periods with strategy-specific restrictions
- **Performance Optimized**: Memory-efficient processing with intelligent caching

### Concrete Trading Recommendations
- **BUY/HOLD/SELL Signals**: Clear actionable recommendations with confidence percentages
- **Market Regime Detection**: Automatic BULL/BEAR/SIDEWAYS market identification
- **Adaptive Thresholds**: Dynamic buy/sell thresholds that adjust to market conditions
- **Risk Management**: Regime-specific stop-loss and take-profit recommendations
- **Backtester-Proven Logic**: Uses the same algorithms validated in historical testing

### Comprehensive Backtesting System
- **Historical Analysis**: Test strategies against years of Alpaca market data
- **Realistic Simulation**: Day-by-day execution with no look-ahead bias
- **Advanced Risk Management**: Adaptive stop-loss, take-profit, and trailing stops based on market regime
- **Market Regime Detection**: Automatic adaptation to BULL/BEAR/SIDEWAYS markets with volatility tolerance
- **Performance Analytics**: Fair buy-and-hold comparison (always starts Day 1) with alpha calculation
- **Bull Market Optimization**: Volatility tolerance system for maximum bull market participation
- **CLI Interface**: Streamlined command-line backtesting with comprehensive batch testing

## 📊 Technical Specifications

### Shared Logic Architecture
The system uses a unified logic module (`src/logic.py`) that ensures consistency between the main program and backtester:

- **Single Source of Truth**: All trading algorithms are centralized in one module
- **Automatic Synchronization**: Changes to logic automatically apply to both systems
- **Proven Algorithms**: Logic is validated through extensive backtesting before deployment
- **Maintainable Code**: No duplication of complex trading algorithms

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
   - **BULL Market**: Wider stop-loss (-40%), higher take-profit (up to 600%), larger positions (up to 3.6x)
   - **STRONG_BULL**: Maximum aggressiveness with bull market duration scaling
   - **BEAR Market**: Tighter stop-loss (-6%), lower take-profit (+12%), smaller positions (0.6x)
   - **SIDEWAYS**: Standard parameters with dynamic threshold adjustment
   - **Volatility Override**: Extreme volatility protection with bull market tolerance

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
7. **Adaptive Risk Management**: Market regime detection with volatility tolerance
8. **Bull Market Duration Scaling**: Progressive aggressiveness in sustained bull markets
9. **Momentum Reversal Detection**: Dynamic regime updates based on price/sentiment divergence

### Source Reliability Tiers
- **PRIMARY** (Weight: 2.0): SEC filings, Company IR
- **INSTITUTIONAL** (Weight: 1.6-1.8): Bloomberg, WSJ, Reuters
- **AGGREGATORS** (Weight: 1.0-1.5): Yahoo Finance, MarketWatch, Benzinga
- **ENTERTAINMENT** (Weight: 0.3-0.8): Motley Fool, Reddit, Social Media

## 🔒 Security & API Keys

**IMPORTANT**: This project requires Alpaca API keys for market data access.

### **Getting API Keys (Free)**
1. Sign up at [Alpaca Markets](https://app.alpaca.markets/signup)
2. Navigate to Paper Trading Dashboard
3. Generate API keys (paper trading is free)
4. Copy `.env.example` to `.env` and add your keys

### **Security Best Practices**
- ✅ API keys are loaded via environment variables (secure)
- ✅ `.env` file is in `.gitignore` (not committed to repo)
- ✅ Uses paper trading endpoint (no real money at risk)
- ⚠️ **Never commit real API keys to version control**

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/orthrus.git
   cd orthrus
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
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your actual Alpaca API keys
   # Get free keys at: https://app.alpaca.markets/paper/dashboard/overview
   ALPACA_API_KEY=your_actual_key_here
   ALPACA_SECRET_KEY=your_actual_secret_here
   ```

## 🎯 Usage

### Real-Time Analysis

#### Interactive Mode (Recommended)
1. Run the program: `python src/main/input.py`
2. Enter a stock ticker (e.g., TSLA, AAPL, MSFT)
3. **Select strategy first** (VALUE or MOMENTUM)
4. Choose timeframe based on strategy restrictions
5. View results and interactive charts with trading signals

#### Headless Mode (CLI/Automation)
1. **Command-line analysis**: `python src/main/headless.py <ticker> <strategy> <period> [--save]`
2. **Examples**:
   ```bash
   # Quick analysis without GUI (run from project root)
   python src/main/headless.py TSLA m 30
   
   # Save chart to file
   python src/main/headless.py AAPL v 90 --save
   ```
3. **Perfect for**: Automation, scripting, server environments, or when you don't want GUI popups

### Historical Backtesting

#### Quick Start
```bash
# Basic backtest
python backtester/run_backtest.py TSLA m 30

# With custom capital and visualization
python backtester/run_backtest.py AAPL v 180 25000 --plot --save
```

#### Command Structure
```bash
python backtester/run_backtest.py <ticker> <strategy> <period> [cash] [--plot] [--save]
```

**Parameters:**
- **ticker**: Stock symbol (TSLA, AAPL, NVDA, MSFT, etc.)
- **strategy**: `v` (VALUE) or `m` (MOMENTUM)
- **period**: Days (30, 90, 180) OR date range (2023-01-20 2023-07-20)
- **cash**: Starting capital (default: $10,000)
- **--plot**: Show performance charts
- **--save**: Save results to JSON file

#### Advanced Testing
```bash
# Random batch testing (new flexible approach)
python backtester/batch_backtest.py           # Run 10 random backtests
python backtester/batch_backtest.py 25        # Run 25 random backtests  
python backtester/batch_backtest.py 50 --seed 123  # Reproducible results

# View all usage examples and tips
python backtester/examples.py
```

### Strategy-Specific Timeframes
- **VALUE Strategy**: 1M, 6M, YTD, MAX (long-term analysis)
- **MOMENTUM Strategy**: 1D, 5D, 1M, 6M (short to medium-term signals)

### Output Interpretation
- **Market Sentiment**: 0-10 scale based on hybrid AI sentiment analysis
- **Market Analysis**: 0-10 based on chosen strategy algorithms
- **Final Buy Score**: Combined score with rolling window analysis
- **Trading Signal**: 🟢 BUY / 🟡 HOLD / 🔴 SELL with confidence percentage
- **Market Regime**: BULL / BEAR / SIDEWAYS detection for adaptive thresholds
- **Adaptive Thresholds**: Dynamic buy/sell thresholds based on market conditions
- **Risk Management**: Regime-specific stop-loss and take-profit recommendations
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
  - Advanced adaptive risk management with market regime detection
  - Bull market volatility tolerance system for enhanced participation
  - Fair buy-and-hold comparison methodology
  - Comprehensive testing framework across multiple market scenarios
  - Performance analysis with detailed categorized results
  - Visual performance charts and P&L tracking

### Quick Reference

#### Basic Command Structure
```bash
python backtester/run_backtest.py <ticker> <strategy> <period> [cash] [--plot] [--save]
```

#### Parameters
- **ticker**: Stock symbol (TSLA, AAPL, NVDA, MSFT, etc.)
- **strategy**: `m` (momentum) or `v` (value)
- **period**: Days (30, 90, 180) OR date range (2023-01-20 2023-07-20)
- **cash**: Starting capital (default: $10,000)
- **--plot**: Show performance charts
- **--save**: Save results to JSON file

#### Usage Examples
```bash
# Basic momentum test (30 days)
python backtester/run_backtest.py TSLA m 30

# Value strategy with custom capital
python backtester/run_backtest.py AAPL v 180 25000

# Historical date range with charts and save
python backtester/run_backtest.py NVDA m 2023-01-20 2023-07-20 --plot --save

# High capital test with full analysis
python backtester/run_backtest.py MSFT m 90 50000 --plot --save

# Quick comparison (momentum vs value)
python backtester/run_backtest.py AAPL m 30 && python backtester/run_backtest.py AAPL v 30

# Comprehensive testing across multiple scenarios
python backtester/batch_backtest.py

# View all usage examples and tips
python backtester/examples.py
```

#### Pro Tips
- **Historical Dates**: Use dates 15+ days old to avoid API rate limits
- **Momentum Strategy**: Works best with 30-90 day periods for swing trading
- **Value Strategy**: Optimal with 90-180 day periods for long-term analysis
- **Validation**: Test multiple tickers to validate strategy robustness
- **Market Conditions**: Test across different market regimes (bull/bear/sideways)
- **Capital Testing**: Try different starting amounts to test scalability
- **Random Testing**: Use batch_backtest.py for diverse, unbiased testing across sectors and time periods

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
