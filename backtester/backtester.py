"""
Alpaca Backtesting System for SigmaStocks
Runs trading strategies against historical Alpaca data with realistic simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys
import os

# Conditional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plotting disabled")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Add parent directory to path to import src modules
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

import config
import market
import analyzer
import scraper
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class AlpacaBacktester:
    """
    Backtesting system that simulates trading strategies using historical Alpaca data
    """
    
    def __init__(self, ticker: str, strategy: str = "momentum", initial_cash: float = 10000.0):
        """
        Initialize the backtester
        
        Args:
            ticker: Stock symbol to backtest
            strategy: Trading strategy ("momentum" or "value")
            initial_cash: Starting cash amount
        """
        self.ticker = ticker.upper()
        self.strategy = strategy.lower()
        self.initial_cash = initial_cash
        
        # Portfolio tracking
        self.cash = initial_cash
        self.shares = 0
        self.portfolio_history = []
        self.trade_log = []
        
        # Risk management
        self.entry_price = None  # Track entry price for stop-loss/take-profit
        self.max_portfolio_value = initial_cash  # Track peak for trailing stop
        
        # Data storage
        self.full_price_data = None
        self.full_news_data = None
        
        # Initialize Alpaca client
        try:
            self.alpaca_client = StockHistoricalDataClient(
                api_key=config.API_KEY,
                secret_key=config.API_SECRET
            )
            print("✅ Alpaca client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Alpaca client: {e}")
            raise
    
    def fetch_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical daily bars from Alpaca
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            print(f"📊 Fetching historical data for {self.ticker} from {start_date.date()} to {end_date.date()}...")
            
            # Create request for daily bars
            request_params = StockBarsRequest(
                symbol_or_symbols=[self.ticker],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            # Fetch data from Alpaca
            bars = self.alpaca_client.get_stock_bars(request_params)
            
            if not bars.df.empty:
                # Convert to standard format
                price_data = bars.df.copy()
                
                # Reset index to get timestamp as column, then set as index
                if 'timestamp' not in price_data.columns:
                    price_data = price_data.reset_index()
                
                price_data.set_index('timestamp', inplace=True)
                
                # Rename columns to match yfinance format
                column_mapping = {
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                price_data.rename(columns=column_mapping, inplace=True)
                
                # Ensure timezone is UTC
                if price_data.index.tz is None:
                    price_data.index = price_data.index.tz_localize('UTC')
                elif price_data.index.tz != pytz.UTC:
                    price_data.index = price_data.index.tz_convert('UTC')
                
                print(f"✅ Fetched {len(price_data)} days of price data")
                return price_data
            else:
                print(f"❌ No price data found for {self.ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def fetch_news_data(self, days: int = 90, end_date: datetime = None) -> List[Tuple]:
        """
        Fetch historical news data using the scraper with support for historical dates
        
        Args:
            days: Number of days of news to fetch
            end_date: End date for news search (for historical backtesting)
            
        Returns:
            List of news tuples (timestamp, headline, source)
        """
        try:
            print(f"📰 Fetching {days} days of news data for {self.ticker}...")
            if end_date:
                print(f"   Historical news ending at: {end_date.date()}")
            
            # Use the existing scraper to get news data with historical support
            profile_data, news_data = scraper.scrape_hybrid(self.ticker, days=days, end_date=end_date)
            
            print(f"✅ Fetched {len(news_data)} news articles")
            return news_data
            
        except Exception as e:
            print(f"❌ Error fetching news data: {e}")
            return []
    
    def prepare_simulation_data(self, start_date: datetime, end_date: datetime, news_days: int = 90):
        """
        Prepare all data needed for simulation
        
        Args:
            start_date: Simulation start date (should be timezone-aware)
            end_date: Simulation end date (should be timezone-aware)
            news_days: Days of news data to fetch (should be >= simulation period)
        """
        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = pytz.utc.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.utc.localize(end_date)
            
        # Fetch price data with some buffer before start_date for technical indicators
        price_start = start_date - timedelta(days=200)  # Buffer for 200-day SMA
        self.full_price_data = self.fetch_historical_data(price_start, end_date)
        
        # Fetch news data with historical end date
        self.full_news_data = self.fetch_news_data(days=news_days, end_date=end_date)
        
        if self.full_price_data.empty:
            raise ValueError(f"No price data available for {self.ticker}")
        
        print(f"📋 Simulation data prepared:")
        print(f"   Price data: {len(self.full_price_data)} records")
        print(f"   News data: {len(self.full_news_data)} articles")
    
    def filter_news_by_date(self, target_date: datetime) -> List[Tuple]:
        """
        Filter news data to only include articles up to target_date
        
        Args:
            target_date: Cut-off date for news
            
        Returns:
            Filtered news list
        """
        if not self.full_news_data:
            return []
        
        # Ensure target_date is timezone-aware (UTC)
        if target_date.tzinfo is None:
            target_date = pytz.utc.localize(target_date)
        else:
            target_date = target_date.astimezone(pytz.utc)
        
        filtered_news = []
        for news_item in self.full_news_data:
            news_date, headline, source = news_item
            
            # Convert news_date to datetime if it's not already
            if isinstance(news_date, str):
                try:
                    news_date = datetime.fromisoformat(news_date.replace('Z', '+00:00'))
                except:
                    continue
            elif not isinstance(news_date, datetime):
                continue
            
            # Ensure timezone awareness for news_date
            if news_date.tzinfo is None:
                news_date = pytz.utc.localize(news_date)
            else:
                news_date = news_date.astimezone(pytz.utc)
            
            # Include news up to target_date
            if news_date <= target_date:
                filtered_news.append((news_date, headline, source))
        
        return filtered_news
    
    def execute_trade(self, action: str, price: float, date: datetime, buy_score: float):
        """
        Execute a buy or sell trade with risk management
        
        Args:
            action: "BUY" or "SELL"
            price: Current stock price
            date: Trade date
            buy_score: The buy score that triggered the trade
        """
        if action == "BUY" and self.cash > 0:
            # Buy as many shares as possible
            shares_to_buy = int(self.cash / price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                self.cash -= cost
                self.shares += shares_to_buy
                self.entry_price = price  # Track entry price for risk management
                
                self.trade_log.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Shares': shares_to_buy,
                    'Price': price,
                    'Cost': cost,
                    'Buy_Score': buy_score,
                    'Cash_After': self.cash,
                    'Shares_After': self.shares
                })
                
                print(f"🟢 BUY: {shares_to_buy} shares at ${price:.2f} (Score: {buy_score:.1f}) - Cash: ${self.cash:.2f}")
        
        elif action == "SELL" and self.shares > 0:
            # Sell all shares
            proceeds = self.shares * price
            self.cash += proceeds
            
            # Calculate P&L for this trade
            if self.entry_price:
                pnl_pct = ((price - self.entry_price) / self.entry_price) * 100
                pnl_dollar = (price - self.entry_price) * self.shares
            else:
                pnl_pct = 0
                pnl_dollar = 0
            
            self.trade_log.append({
                'Date': date,
                'Action': 'SELL',
                'Shares': self.shares,
                'Price': price,
                'Proceeds': proceeds,
                'Buy_Score': buy_score,
                'Cash_After': self.cash,
                'Shares_After': 0,
                'PnL_Pct': pnl_pct,
                'PnL_Dollar': pnl_dollar
            })
            
            print(f"🔴 SELL: {self.shares} shares at ${price:.2f} (Score: {buy_score:.1f}) - P&L: {pnl_pct:+.1f}% (${pnl_dollar:+.0f}) - Cash: ${self.cash:.2f}")
            self.shares = 0
            self.entry_price = None  # Reset entry price
    
    def detect_market_regime(self, price_data: pd.DataFrame, lookback_days: int = 60) -> str:
        """
        Detect current market regime (BULL, BEAR, SIDEWAYS) for adaptive strategy
        
        Args:
            price_data: Historical price data
            lookback_days: Days to analyze for regime detection
            
        Returns:
            Market regime: "BULL", "BEAR", or "SIDEWAYS"
        """
        if len(price_data) < lookback_days:
            return "SIDEWAYS"  # Default if not enough data
        
        recent_data = price_data.tail(lookback_days)
        prices = recent_data['Close']
        
        # Calculate trend metrics
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        total_return = (end_price - start_price) / start_price
        
        # Calculate moving averages for trend confirmation
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else sma_20
        
        # Calculate volatility (standard deviation of returns)
        returns = prices.pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
        
        # Regime detection logic
        if total_return > 0.15 and end_price > sma_20 > sma_50 and volatility < 0.6:
            return "BULL"  # Strong uptrend with low volatility
        elif total_return < -0.15 and end_price < sma_20 < sma_50:
            return "BEAR"  # Strong downtrend
        else:
            return "SIDEWAYS"  # Choppy or neutral market
    
    def get_adaptive_risk_params(self, market_regime: str) -> Dict:
        """
        Get adaptive risk management parameters based on market regime
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary with risk parameters
        """
        if market_regime == "BULL":
            return {
                'stop_loss_pct': -0.12,      # Wider stop-loss (12% vs 8%)
                'take_profit_pct': 0.40,     # Higher take-profit (40% vs 25%)
                'trailing_stop_pct': -0.15,  # Wider trailing stop (15% vs 10%)
                'position_size_multiplier': 1.2,  # Larger positions
                'threshold_tightness': 0.7    # Tighter thresholds (more trades)
            }
        elif market_regime == "BEAR":
            return {
                'stop_loss_pct': -0.06,      # Tighter stop-loss (6% vs 8%)
                'take_profit_pct': 0.15,     # Lower take-profit (15% vs 25%)
                'trailing_stop_pct': -0.08,  # Tighter trailing stop (8% vs 10%)
                'position_size_multiplier': 0.8,  # Smaller positions
                'threshold_tightness': 1.3    # Wider thresholds (fewer trades)
            }
        else:  # SIDEWAYS
            return {
                'stop_loss_pct': -0.08,      # Standard parameters
                'take_profit_pct': 0.25,
                'trailing_stop_pct': -0.10,
                'position_size_multiplier': 1.0,
                'threshold_tightness': 1.0
            }
    
    def calculate_adaptive_thresholds(self, score_history: List[float], market_regime: str, lookback: int = 20) -> Tuple[float, float]:
        """
        Calculate adaptive buy/sell thresholds based on market regime and score volatility
        
        Args:
            score_history: List of recent Final_Buy_Scores
            market_regime: Current market regime
            lookback: Number of recent scores to analyze
            
        Returns:
            Tuple of (buy_threshold, sell_threshold)
        """
        if len(score_history) < lookback:
            # Not enough history, use regime-based defaults
            if market_regime == "BULL":
                return 5.01, 4.99  # Tighter thresholds for more trades
            elif market_regime == "BEAR":
                return 5.15, 4.85  # Wider thresholds for selectivity
            else:
                return 5.02, 4.98  # Standard thresholds
        
        recent_scores = score_history[-lookback:]
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        # Get regime-specific parameters
        risk_params = self.get_adaptive_risk_params(market_regime)
        tightness = risk_params['threshold_tightness']
        
        # Adaptive volatility factor
        base_volatility_factor = min(std_score * 2, 0.5)
        adjusted_volatility_factor = base_volatility_factor * tightness
        
        # Calculate thresholds
        buy_threshold = mean_score + adjusted_volatility_factor
        sell_threshold = mean_score - adjusted_volatility_factor
        
        # Regime-specific bounds
        if market_regime == "BULL":
            buy_threshold = max(5.005, min(buy_threshold, 5.5))   # Tighter bounds
            sell_threshold = max(4.5, min(sell_threshold, 4.995))
        elif market_regime == "BEAR":
            buy_threshold = max(5.05, min(buy_threshold, 6.0))    # Wider bounds
            sell_threshold = max(4.0, min(sell_threshold, 4.95))
        else:  # SIDEWAYS
            buy_threshold = max(5.01, min(buy_threshold, 5.8))
            sell_threshold = max(4.2, min(sell_threshold, 4.99))
        
        return buy_threshold, sell_threshold
    
    def check_adaptive_risk_management(self, current_price: float, portfolio_value: float, market_regime: str) -> Tuple[bool, str]:
        """
        Check if adaptive risk management rules trigger a sell based on market regime
        
        Args:
            current_price: Current stock price
            portfolio_value: Current portfolio value
            market_regime: Current market regime
            
        Returns:
            Tuple of (should_sell, reason)
        """
        if self.shares == 0 or self.entry_price is None:
            return False, ""
        
        # Get adaptive risk parameters
        risk_params = self.get_adaptive_risk_params(market_regime)
        
        # Update max portfolio value for trailing stop
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        # Calculate current P&L
        current_pnl_pct = ((current_price - self.entry_price) / self.entry_price)
        
        # Adaptive stop-loss
        if current_pnl_pct <= risk_params['stop_loss_pct']:
            return True, f"STOP-LOSS ({risk_params['stop_loss_pct']*100:.0f}%): {current_pnl_pct*100:.1f}%"
        
        # Adaptive take-profit
        if current_pnl_pct >= risk_params['take_profit_pct']:
            return True, f"TAKE-PROFIT (+{risk_params['take_profit_pct']*100:.0f}%): {current_pnl_pct*100:.1f}%"
        
        # Adaptive trailing stop
        trailing_stop_pct = ((portfolio_value - self.max_portfolio_value) / self.max_portfolio_value)
        if trailing_stop_pct <= risk_params['trailing_stop_pct']:
            return True, f"TRAILING-STOP ({risk_params['trailing_stop_pct']*100:.0f}% from peak): {trailing_stop_pct*100:.1f}%"
        
        return False, ""
    
    def execute_adaptive_trade(self, action: str, price: float, date: datetime, buy_score: float, market_regime: str):
        """
        Execute a buy or sell trade with adaptive position sizing based on market regime
        
        Args:
            action: "BUY" or "SELL"
            price: Current stock price
            date: Trade date
            buy_score: The buy score that triggered the trade
            market_regime: Current market regime
        """
        if action == "BUY" and self.cash > 0:
            # Get adaptive position sizing
            risk_params = self.get_adaptive_risk_params(market_regime)
            position_multiplier = risk_params['position_size_multiplier']
            
            # Calculate adaptive position size
            base_shares = int(self.cash / price)
            adjusted_shares = int(base_shares * position_multiplier)
            shares_to_buy = min(adjusted_shares, base_shares)  # Don't exceed available cash
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                self.cash -= cost
                self.shares += shares_to_buy
                self.entry_price = price  # Track entry price for risk management
                
                self.trade_log.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Shares': shares_to_buy,
                    'Price': price,
                    'Cost': cost,
                    'Buy_Score': buy_score,
                    'Market_Regime': market_regime,
                    'Position_Multiplier': position_multiplier,
                    'Cash_After': self.cash,
                    'Shares_After': self.shares
                })
                
                print(f"🟢 BUY: {shares_to_buy} shares at ${price:.2f} (Score: {buy_score:.1f}, {market_regime}) - Cash: ${self.cash:.2f}")
        
        elif action == "SELL" and self.shares > 0:
            # Sell all shares
            proceeds = self.shares * price
            self.cash += proceeds
            
            # Calculate P&L for this trade
            if self.entry_price:
                pnl_pct = ((price - self.entry_price) / self.entry_price) * 100
                pnl_dollar = (price - self.entry_price) * self.shares
            else:
                pnl_pct = 0
                pnl_dollar = 0
            
            self.trade_log.append({
                'Date': date,
                'Action': 'SELL',
                'Shares': self.shares,
                'Price': price,
                'Proceeds': proceeds,
                'Buy_Score': buy_score,
                'Market_Regime': market_regime,
                'Cash_After': self.cash,
                'Shares_After': 0,
                'PnL_Pct': pnl_pct,
                'PnL_Dollar': pnl_dollar
            })
            
            print(f"🔴 SELL: {self.shares} shares at ${price:.2f} (Score: {buy_score:.1f}, {market_regime}) - P&L: {pnl_pct:+.1f}% (${pnl_dollar:+.0f}) - Cash: ${self.cash:.2f}")
            self.shares = 0
            self.entry_price = None  # Reset entry price
    
    def run(self, start_date: datetime, end_date: datetime, lookback_days: int = 30) -> Dict:
        """
        Run the backtesting simulation with fully optimized pre-calculated scores
        
        Args:
            start_date: Simulation start date
            end_date: Simulation end date
            lookback_days: Days to look back for analysis
            
        Returns:
            Dictionary with simulation results
        """
        print(f"\n🚀 Starting backtest simulation for {self.ticker}")
        print(f"📅 Period: {start_date.date()} to {end_date.date()}")
        print(f"💰 Initial cash: ${self.initial_cash:,.2f}")
        print(f"📈 Strategy: {self.strategy.upper()}")
        print("=" * 60)
        
        # Ensure start_date and end_date are timezone-aware (UTC)
        if start_date.tzinfo is None:
            start_date = pytz.utc.localize(start_date)
        else:
            start_date = start_date.astimezone(pytz.utc)
            
        if end_date.tzinfo is None:
            end_date = pytz.utc.localize(end_date)
        else:
            end_date = end_date.astimezone(pytz.utc)
        
        # Prepare simulation data
        news_days = max(90, (end_date - start_date).days + 30)  # Ensure enough news data
        self.prepare_simulation_data(start_date, end_date, news_days)
        
        # Get trading days from price data
        trading_days = self.full_price_data[
            (self.full_price_data.index >= start_date) & 
            (self.full_price_data.index <= end_date)
        ].index
        
        if len(trading_days) == 0:
            raise ValueError("No trading days found in the specified period")
        
        print(f"📊 Simulating {len(trading_days)} trading days...")
        
        # OPTIMIZATION: Analyze ALL news articles ONCE and calculate ALL scores ONCE
        print(f"🧠 Analyzing all {len(self.full_news_data)} news articles (one-time processing)...")
        try:
            # Analyze all news at once to get complete sentiment DataFrame
            full_sentiment_df = analyzer.get_sentiment(self.full_news_data, self.ticker, lookback_days)
            print(f"✅ Sentiment analysis complete: {len(full_sentiment_df)} sentiment records")
        except Exception as e:
            print(f"❌ Error in sentiment analysis: {e}")
            full_sentiment_df = pd.DataFrame()
        
        if full_sentiment_df.empty:
            print("⚠️  No sentiment data available, using neutral scores")
            # Create a simple lookup with neutral scores for all trading days
            score_lookup = {date: 5.0 for date in trading_days}
        else:
            # PRE-CALCULATE ALL Final_Buy_Scores using the complete dataset
            print(f"🔢 Pre-calculating all Final_Buy_Scores for entire period...")
            
            # Calculate verdict using the FULL sentiment and price data
            verdict = market.calculate_verdict(
                ticker=self.ticker,
                sentiment_df=full_sentiment_df,
                strategy=self.strategy,
                lookback_days=lookback_days,
                custom_date=end_date,  # Use end date as reference
                price_data=self.full_price_data  # Use full price data
            )
            
            # Extract all Final_Buy_Scores and create a lookup dictionary
            final_buy_scores_over_time = verdict.get('Final_Buy_Scores_Over_Time', [])
            score_lookup = {}
            
            for timestamp, score in final_buy_scores_over_time:
                # Create lookup by date for fast access
                if hasattr(timestamp, 'date'):
                    date_key = timestamp.date()
                else:
                    date_key = timestamp
                score_lookup[date_key] = score
            
            print(f"✅ Pre-calculated {len(score_lookup)} Final_Buy_Scores")
        
        print()
        
        # MARKET REGIME DETECTION for adaptive strategy
        market_regime = self.detect_market_regime(self.full_price_data)
        print(f"🎯 Market Regime Detected: {market_regime}")
        
        risk_params = self.get_adaptive_risk_params(market_regime)
        print(f"📊 Adaptive Parameters: Stop-loss: {risk_params['stop_loss_pct']*100:.0f}%, Take-profit: {risk_params['take_profit_pct']*100:.0f}%, Position size: {risk_params['position_size_multiplier']:.1f}x")
        print()
        
        # Simulation loop - now extremely fast since everything is pre-calculated
        score_history = []  # Track scores for dynamic thresholds
        
        for i, simulation_date in enumerate(trading_days):
            try:
                current_price = self.full_price_data.loc[simulation_date, 'Close']
                
                # FAST LOOKUP: Get pre-calculated Final_Buy_Score for this date
                sim_date_key = simulation_date.date() if hasattr(simulation_date, 'date') else simulation_date
                
                # Look for exact match first
                if sim_date_key in score_lookup:
                    buy_score_t = score_lookup[sim_date_key]
                else:
                    # Find the most recent score on or before this date
                    valid_dates = [d for d in score_lookup.keys() if d <= sim_date_key]
                    if valid_dates:
                        most_recent_date = max(valid_dates)
                        buy_score_t = score_lookup[most_recent_date]
                    else:
                        buy_score_t = 5.0  # Neutral fallback
                
                # Add to score history for dynamic thresholds
                score_history.append(buy_score_t)
                
                # Calculate portfolio value
                portfolio_value = self.cash + (self.shares * current_price)
                
                # ADAPTIVE RISK MANAGEMENT: Check regime-specific stop-loss/take-profit first
                should_sell_risk, risk_reason = self.check_adaptive_risk_management(current_price, portfolio_value, market_regime)
                if should_sell_risk:
                    self.execute_adaptive_trade("SELL", current_price, simulation_date, buy_score_t, market_regime)
                    print(f"🛡️  {risk_reason}")
                else:
                    # ADAPTIVE THRESHOLDS: Calculate regime-aware buy/sell thresholds
                    buy_threshold, sell_threshold = self.calculate_adaptive_thresholds(score_history, market_regime)
                    
                    # Debug: Print adaptive thresholds for first few days
                    if i < 3:
                        print(f"📊 Day {i+1}: Final_Buy_Score = {buy_score_t:.2f} | {market_regime} Thresholds: Buy≥{buy_threshold:.3f}, Sell≤{sell_threshold:.3f}")
                    
                    # Trading logic using adaptive thresholds
                    if buy_score_t >= buy_threshold and self.shares == 0:
                        # Strong buy signal - enter position with adaptive sizing
                        self.execute_adaptive_trade("BUY", current_price, simulation_date, buy_score_t, market_regime)
                    elif buy_score_t <= sell_threshold and self.shares > 0:
                        # Weak signal - exit position
                        self.execute_adaptive_trade("SELL", current_price, simulation_date, buy_score_t, market_regime)
                
                # Track portfolio value
                self.portfolio_history.append({
                    'Date': simulation_date,
                    'Price': current_price,
                    'Cash': self.cash,
                    'Shares': self.shares,
                    'Portfolio_Value': portfolio_value,
                    'Final_Buy_Score': buy_score_t,  # Use pre-calculated Final_Buy_Score
                    'Market_Regime': market_regime
                })
                
                # Progress update every 10 days with buy score info and adaptive thresholds
                if (i + 1) % 10 == 0:
                    buy_thresh, sell_thresh = self.calculate_adaptive_thresholds(score_history, market_regime)
                    print(f"📈 Day {i+1}/{len(trading_days)}: ${portfolio_value:,.2f} (Score: {buy_score_t:.1f}, {market_regime}) | Thresholds: {buy_thresh:.3f}/{sell_thresh:.3f}")
            
            except Exception as e:
                print(f"⚠️  Error on {simulation_date.date()}: {e}")
                continue
        
        # Final portfolio value
        final_price = self.portfolio_history[-1]['Price'] if self.portfolio_history else 0
        final_portfolio_value = self.cash + (self.shares * final_price)
        
        # Calculate performance metrics
        total_return = ((final_portfolio_value - self.initial_cash) / self.initial_cash) * 100
        
        # Buy and hold comparison
        initial_price = self.full_price_data.loc[trading_days[0], 'Close']
        buy_hold_shares = self.initial_cash / initial_price
        buy_hold_value = buy_hold_shares * final_price
        buy_hold_return = ((buy_hold_value - self.initial_cash) / self.initial_cash) * 100
        
        results = {
            'ticker': self.ticker,
            'strategy': self.strategy,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': self.initial_cash,
            'final_portfolio_value': final_portfolio_value,
            'final_cash': self.cash,
            'final_shares': self.shares,
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return,
            'alpha': total_return - buy_hold_return,
            'num_trades': len(self.trade_log),
            'portfolio_history': self.portfolio_history,
            'trade_log': self.trade_log
        }
        
        self.print_results(results)
        return results
    
    def print_results(self, results: Dict):
        """Print backtesting results summary"""
        print("\n" + "=" * 60)
        print("📊 BACKTESTING RESULTS")
        print("=" * 60)
        print(f"Ticker: {results['ticker']}")
        print(f"Strategy: {results['strategy'].upper()}")
        print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
        print(f"Initial Cash: ${results['initial_cash']:,.2f}")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Final Cash: ${results['final_cash']:,.2f}")
        print(f"Final Shares: {results['final_shares']}")
        print()
        print(f"📈 Strategy Return: {results['total_return_pct']:+.2f}%")
        print(f"📊 Buy & Hold Return: {results['buy_hold_return_pct']:+.2f}%")
        print(f"🎯 Alpha (Outperformance): {results['alpha']:+.2f}%")
        print(f"🔄 Number of Trades: {results['num_trades']}")
        
        # Add buy score statistics
        if results['portfolio_history']:
            buy_scores = [h['Final_Buy_Score'] for h in results['portfolio_history']]
            print()
            print(f"📊 Buy Score Stats:")
            print(f"   Max: {max(buy_scores):.2f}")
            print(f"   Min: {min(buy_scores):.2f}")
            print(f"   Avg: {sum(buy_scores)/len(buy_scores):.2f}")
            print(f"   Days ≥5.02 (Buy): {sum(1 for s in buy_scores if s >= 5.02)}")
            print(f"   Days ≤4.98 (Sell): {sum(1 for s in buy_scores if s <= 4.98)}")
        
        print()
        
        if results['trade_log']:
            print("📋 TRADE LOG:")
            for trade in results['trade_log']:
                action_emoji = "🟢" if trade['Action'] == 'BUY' else "🔴"
                regime = trade.get('Market_Regime', 'N/A')
                if trade['Action'] == 'SELL' and 'PnL_Pct' in trade:
                    pnl_info = f" | P&L: {trade['PnL_Pct']:+.1f}%"
                else:
                    pnl_info = ""
                print(f"{action_emoji} {trade['Date'].date()} | {trade['Action']} {trade['Shares']} @ ${trade['Price']:.2f} | Score: {trade['Buy_Score']:.1f} ({regime}){pnl_info}")
        
        print("=" * 60)
    
    def plot_results(self, results: Dict):
        """Plot portfolio performance vs stock price"""
        if not results['portfolio_history']:
            print("No portfolio history to plot")
            return
        
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return
        
        # Check if we're in a headless environment or batch mode
        import os
        if os.environ.get('DISPLAY') is None and os.name != 'nt':
            print("No display available for plotting (headless environment)")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(results['portfolio_history'])
        df.set_index('Date', inplace=True)
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'{results["ticker"]} Backtesting Results - {results["strategy"].upper()} Strategy', fontsize=16)
        
        # Plot 1: Portfolio Value vs Stock Price
        ax1_twin = ax1.twinx()
        
        # Portfolio value (left axis)
        ax1.plot(df.index, df['Portfolio_Value'], 'b-', linewidth=2, label='Portfolio Value')
        ax1.set_ylabel('Portfolio Value ($)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Stock price (right axis)
        ax1_twin.plot(df.index, df['Price'], 'r-', linewidth=1, alpha=0.7, label='Stock Price')
        ax1_twin.set_ylabel('Stock Price ($)', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # Add trade markers
        if results['trade_log']:
            for trade in results['trade_log']:
                color = 'green' if trade['Action'] == 'BUY' else 'red'
                marker = '^' if trade['Action'] == 'BUY' else 'v'
                ax1.scatter(trade['Date'], df.loc[trade['Date'], 'Portfolio_Value'], 
                           color=color, marker=marker, s=100, zorder=5)
        
        ax1.set_title('Portfolio Value vs Stock Price')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: Buy Scores
        ax2.plot(df.index, df['Final_Buy_Score'], 'g-', linewidth=1)
        ax2.axhline(y=5.02, color='green', linestyle='--', alpha=0.7, label='Buy Threshold (5.02)')
        ax2.axhline(y=4.98, color='red', linestyle='--', alpha=0.7, label='Sell Threshold (4.98)')
        ax2.axhline(y=5.0, color='gray', linestyle='-', alpha=0.5, label='Neutral (5.0)')
        ax2.set_ylabel('Buy Score')
        ax2.set_title('Buy Score Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 10)
        
        # Plot 3: Cash and Shares
        ax3_twin = ax3.twinx()
        
        # Cash (left axis)
        ax3.plot(df.index, df['Cash'], 'b-', linewidth=2, label='Cash')
        ax3.set_ylabel('Cash ($)', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        
        # Shares (right axis)
        ax3_twin.plot(df.index, df['Shares'], 'orange', linewidth=2, label='Shares')
        ax3_twin.set_ylabel('Shares Held', color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        ax3.set_title('Cash and Shares Over Time')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
            print("Plot generated but display failed (possibly headless environment)")
        
        # Print summary statistics
        print(f"\n📊 Performance Summary:")
        print(f"Max Portfolio Value: ${df['Portfolio_Value'].max():,.2f}")
        print(f"Min Portfolio Value: ${df['Portfolio_Value'].min():,.2f}")
        print(f"Volatility (Portfolio): {df['Portfolio_Value'].pct_change().std() * 100:.2f}%")
        print(f"Volatility (Stock): {df['Price'].pct_change().std() * 100:.2f}%")


def run_backtest_example():
    """Example function showing how to use the backtester"""
    
    # Configuration
    ticker = "TSLA"
    strategy = "momentum"  # or "value"
    initial_cash = 10000.0
    
    # Date range (last 3 months for example)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    try:
        # Initialize backtester
        backtester = AlpacaBacktester(ticker, strategy, initial_cash)
        
        # Run simulation
        results = backtester.run(start_date, end_date, lookback_days=30)
        
        # Plot results
        backtester.plot_results(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        return None


if __name__ == "__main__":
    # Run example backtest
    run_backtest_example()