"""
Shared Logic Module
Contains market regime detection, adaptive thresholds, and risk management logic
Used by both main program and backtester to ensure consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yfinance as yf


def detect_market_regime(ticker: str = None, price_data: pd.DataFrame = None, lookback_days: int = 60) -> str:
    """
    Detect current market regime (BULL, BEAR, SIDEWAYS) for adaptive strategy
    
    Args:
        ticker: Stock symbol (if price_data not provided)
        price_data: Historical price data (if ticker not provided)
        lookback_days: Days to analyze for regime detection
        
    Returns:
        Market regime: "BULL", "BEAR", or "SIDEWAYS"
    """
    try:
        # Get price data if not provided
        if price_data is None:
            if ticker is None:
                return "SIDEWAYS"
            stock = yf.Ticker(ticker)
            price_data = stock.history(period=f"{lookback_days + 10}d")
        
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
            
    except Exception as e:
        print(f"⚠️  Market regime detection failed: {e}")
        return "SIDEWAYS"  # Safe default


def get_adaptive_risk_params(market_regime: str) -> Dict:
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


def calculate_adaptive_thresholds(score_history: List[float], market_regime: str, lookback: int = 20) -> Tuple[float, float]:
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
    risk_params = get_adaptive_risk_params(market_regime)
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


def get_trading_recommendation(ticker: str, final_buy_score: float, sentiment_df: pd.DataFrame = None) -> Dict:
    """
    Convert Final_Buy_Score into concrete BUY/HOLD/SELL recommendation
    Uses adaptive thresholds and market regime detection
    
    Args:
        ticker: Stock symbol
        final_buy_score: Current Final_Buy_Score
        sentiment_df: Historical sentiment data for score history (optional)
        
    Returns:
        Dictionary with trading recommendation and details
    """
    # Detect current market regime
    market_regime = detect_market_regime(ticker)
    
    # Get historical Final_Buy_Scores if available
    score_history = []
    if sentiment_df is not None and hasattr(sentiment_df, 'attrs') and 'Final_Buy_Scores_Over_Time' in sentiment_df.attrs:
        scores_over_time = sentiment_df.attrs['Final_Buy_Scores_Over_Time']
        score_history = [score for _, score in scores_over_time]
    else:
        # Fallback: use current score repeated (not ideal but functional)
        score_history = [final_buy_score] * 10
    
    # Calculate adaptive thresholds
    buy_threshold, sell_threshold = calculate_adaptive_thresholds(score_history, market_regime)
    
    # Get risk parameters
    risk_params = get_adaptive_risk_params(market_regime)
    
    # Determine recommendation
    if final_buy_score >= buy_threshold:
        recommendation = "BUY"
        confidence = min(100, ((final_buy_score - buy_threshold) / (10 - buy_threshold)) * 100)
        reasoning = f"Score {final_buy_score:.2f} exceeds {market_regime} buy threshold {buy_threshold:.3f}"
    elif final_buy_score <= sell_threshold:
        recommendation = "SELL"
        confidence = min(100, ((sell_threshold - final_buy_score) / sell_threshold) * 100)
        reasoning = f"Score {final_buy_score:.2f} below {market_regime} sell threshold {sell_threshold:.3f}"
    else:
        recommendation = "HOLD"
        confidence = 100 - abs(final_buy_score - 5.0) * 20  # Closer to neutral = higher confidence
        reasoning = f"Score {final_buy_score:.2f} within {market_regime} hold range ({sell_threshold:.3f} - {buy_threshold:.3f})"
    
    return {
        'recommendation': recommendation,
        'confidence': max(0, min(100, confidence)),
        'market_regime': market_regime,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'reasoning': reasoning,
        'risk_params': risk_params,
        'final_buy_score': final_buy_score
    }