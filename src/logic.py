"""
Shared Logic Module
Contains market regime detection, adaptive thresholds, and risk management logic
Used by both main program and backtester to ensure consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yfinance as yf


def calculate_bull_market_duration(price_data: pd.DataFrame, current_regime: str, lookback_days: int = 120) -> int:
    """
    Calculate how long the current bull market has been running
    
    Args:
        price_data: Historical price data
        current_regime: Current market regime
        lookback_days: Days to look back for regime analysis
        
    Returns:
        Number of days in current bull market (0 if not in bull market)
    """
    if current_regime not in ["BULL", "STRONG_BULL"]:
        return 0
    
    try:
        if len(price_data) < lookback_days:
            return 0
        
        recent_data = price_data.tail(lookback_days)
        prices = recent_data['Close']
        
        # Calculate rolling 20-day returns to identify bull market start
        rolling_returns = prices.pct_change(20).dropna()
        
        # Find the start of the current bull market (first day with sustained positive momentum)
        bull_start_idx = None
        for i in range(len(rolling_returns) - 1, -1, -1):
            if rolling_returns.iloc[i] < 0.05:  # Less than 5% 20-day return
                bull_start_idx = i + 1
                break
        
        if bull_start_idx is None:
            # Entire lookback period is bull market
            return lookback_days
        else:
            # Calculate days since bull market started
            return len(rolling_returns) - bull_start_idx
            
    except Exception as e:
        print(f"⚠️  Bull market duration calculation failed: {e}")
        return 0


def calculate_adaptive_profit_target(entry_price: float, current_price: float, bull_market_duration: int, 
                                   base_profit_pct: float, market_regime: str) -> float:
    """
    Calculate adaptive profit target that scales with bull market momentum and duration
    
    Args:
        entry_price: Original entry price
        current_price: Current stock price
        bull_market_duration: Days in current bull market
        base_profit_pct: Base profit target percentage
        market_regime: Current market regime
        
    Returns:
        Adaptive profit target percentage
    """
    if market_regime not in ["BULL", "STRONG_BULL"]:
        return base_profit_pct
    
    # Calculate current unrealized gain
    current_gain_pct = (current_price - entry_price) / entry_price
    
    # MOMENTUM SCALING: Higher targets for sustained bull markets
    duration_multiplier = min(1.0 + (bull_market_duration / 60.0), 2.5)  # Up to 2.5x over 60 days
    
    # MOMENTUM PERSISTENCE: If already profitable, scale target higher
    if current_gain_pct > 0.1:  # If already up 10%
        momentum_multiplier = min(1.0 + (current_gain_pct * 2), 2.0)  # Up to 2x based on current gains
    else:
        momentum_multiplier = 1.0
    
    # REGIME SCALING: Different scaling for different bull market strengths
    if market_regime == "STRONG_BULL":
        regime_multiplier = 1.5
    else:  # BULL
        regime_multiplier = 1.2
    
    # Calculate adaptive target
    adaptive_target = base_profit_pct * duration_multiplier * momentum_multiplier * regime_multiplier
    
    # Cap the target to prevent excessive greed
    max_target = 3.0 if market_regime == "STRONG_BULL" else 2.0  # 300% or 200% max
    adaptive_target = min(adaptive_target, max_target)
    
    return adaptive_target


def detect_market_regime(ticker: str = None, price_data: pd.DataFrame = None, lookback_days: int = 60) -> str:
    """
    Detect current market regime (BULL, BEAR, SIDEWAYS) for adaptive strategy
    Enhanced with price volatility protection to prevent overtrading in volatile conditions
    
    Args:
        ticker: Stock symbol (if price_data not provided)
        price_data: Historical price data (if ticker not provided)
        lookback_days: Days to analyze for regime detection
        
    Returns:
        Market regime: "BULL", "STRONG_BULL", "BEAR", "STRONG_BEAR", or "SIDEWAYS"
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
        
        # Calculate momentum indicators
        recent_momentum = (prices.iloc[-10:].mean() - prices.iloc[-20:-10].mean()) / prices.iloc[-20:-10].mean() if len(prices) >= 20 else 0
        
        # VOLATILITY OVERRIDE: Extreme price volatility should force SIDEWAYS classification
        extreme_volatility_threshold = 0.8  # 80% annualized volatility (lowered from 120%)
        if volatility > extreme_volatility_threshold:
            print(f"⚠️  EXTREME PRICE VOLATILITY DETECTED: {volatility:.1%} annualized - forcing SIDEWAYS regime to prevent overtrading")
            return "SIDEWAYS"
        
        # More aggressive bull market detection for maximum participation
        if total_return > 0.20 and end_price > sma_20 > sma_50 and recent_momentum > 0.05 and volatility < 0.9:
            return "STRONG_BULL"  # Strong uptrend with momentum and reasonable volatility
        elif total_return > 0.10 and end_price > sma_20 and recent_momentum > 0.02 and volatility < 1.0:
            return "BULL"  # Uptrend with momentum and acceptable volatility
        elif total_return > 0.05 and end_price > sma_20 and volatility < 0.7:
            return "BULL"  # Moderate uptrend with low volatility - still bullish
        elif total_return < -0.20 and end_price < sma_20 < sma_50:
            return "STRONG_BEAR"  # Strong downtrend
        elif total_return < -0.10 and end_price < sma_20:
            return "BEAR"  # Downtrend
        else:
            # Check for hidden bull momentum even in sideways markets (but not if too volatile)
            if recent_momentum > 0.03 and volatility < 0.8:
                return "BULL"  # Momentum override - treat as bull market
            return "SIDEWAYS"  # Neutral market
            
    except Exception as e:
        print(f"⚠️  Market regime detection failed: {e}")
        return "SIDEWAYS"  # Safe default


def calculate_price_volatility(ticker: str = None, price_data: pd.DataFrame = None, lookback_days: int = 30) -> float:
    """
    Calculate annualized price volatility for dynamic risk adjustment
    
    Args:
        ticker: Stock symbol (if price_data not provided)
        price_data: Historical price data (if ticker not provided)
        lookback_days: Days to analyze for volatility calculation
        
    Returns:
        Annualized volatility as decimal (e.g., 0.5 = 50%)
    """
    try:
        # Get price data if not provided
        if price_data is None:
            if ticker is None:
                return 0.3  # Default moderate volatility
            stock = yf.Ticker(ticker)
            price_data = stock.history(period=f"{lookback_days + 5}d")
        
        if len(price_data) < lookback_days:
            return 0.3  # Default if not enough data
        
        recent_data = price_data.tail(lookback_days)
        prices = recent_data['Close']
        
        # Calculate daily returns and annualized volatility
        returns = prices.pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
        
        return max(0.1, min(volatility, 3.0))  # Clamp between 10% and 300%
        
    except Exception as e:
        print(f"⚠️  Price volatility calculation failed: {e}")
        return 0.3  # Safe default


def get_adaptive_risk_params(market_regime: str, price_volatility: float = None, strategy: str = "momentum", bull_market_duration: int = 0) -> Dict:
    """
    Get adaptive risk management parameters based on market regime, price volatility, and strategy
    Enhanced with volatility-adjusted risk management, strategy-specific optimizations, and bull market duration scaling
    
    Args:
        market_regime: Current market regime
        price_volatility: Annualized price volatility (optional, will calculate if not provided)
        strategy: Trading strategy ("momentum" or "value") for strategy-specific optimizations
        bull_market_duration: Days in current bull market (for scaling bull market aggressiveness)
        
    Returns:
        Dictionary with risk parameters adjusted for volatility, strategy, and bull market duration
    """
    # Default volatility if not provided
    if price_volatility is None:
        price_volatility = 0.3  # 30% default
    
    # VOLATILITY ADJUSTMENT FACTORS
    # Base multipliers for different volatility levels
    if price_volatility > 1.5:  # Extreme volatility (>150%)
        vol_stop_multiplier = 3.0    # Much wider stops
        vol_profit_multiplier = 0.6  # Lower profit targets
        vol_position_multiplier = 0.4  # Much smaller positions
        vol_trail_multiplier = 3.5   # Much wider trailing stops
        print(f"🔥 EXTREME VOLATILITY DETECTED: {price_volatility:.1%} - applying maximum risk protection")
    elif price_volatility > 1.0:  # High volatility (>100%)
        vol_stop_multiplier = 2.0    # Wider stops
        vol_profit_multiplier = 0.7  # Slightly lower profit targets
        vol_position_multiplier = 0.6  # Smaller positions
        vol_trail_multiplier = 2.5   # Wider trailing stops
        print(f"⚠️  HIGH VOLATILITY DETECTED: {price_volatility:.1%} - applying enhanced risk protection")
    elif price_volatility > 0.6:  # Moderate-high volatility (>60%)
        vol_stop_multiplier = 1.5    # Moderately wider stops
        vol_profit_multiplier = 0.85 # Slightly lower profit targets
        vol_position_multiplier = 0.8  # Slightly smaller positions
        vol_trail_multiplier = 1.8   # Moderately wider trailing stops
    else:  # Normal volatility (≤60%)
        vol_stop_multiplier = 1.0    # Standard parameters
        vol_profit_multiplier = 1.0
        vol_position_multiplier = 1.0
        vol_trail_multiplier = 1.0
    
    # VALUE STRATEGY OPTIMIZATIONS: More conservative, longer holding periods
    if strategy.lower() == "value":
        # Value stocks benefit from longer holding periods and more conservative risk management
        value_stop_multiplier = 1.3      # Wider stops for value stocks (less noise)
        value_profit_multiplier = 0.7    # Lower profit targets (value stocks move slower)
        value_position_multiplier = 1.2  # Larger positions (value stocks are safer)
        value_min_hold_multiplier = 3.0  # Much longer minimum holding periods
        value_threshold_multiplier = 1.4 # Wider thresholds (less frequent trading)
        
        # Only print once per function call
        if not hasattr(get_adaptive_risk_params, '_value_msg_shown'):
            print(f"📊 VALUE STRATEGY OPTIMIZATIONS: Wider stops (+30%), longer holds (3x), larger positions (+20%)")
            get_adaptive_risk_params._value_msg_shown = True
    else:
        # Momentum strategy uses standard multipliers
        value_stop_multiplier = 1.0
        value_profit_multiplier = 1.0
        value_position_multiplier = 1.0
        value_min_hold_multiplier = 1.0
        value_threshold_multiplier = 1.0
    
    # Base parameters by regime with moderate bull market duration scaling
    if market_regime == "STRONG_BULL":
        # Moderate bull market scaling for maximum participation
        duration_multiplier = min(1.0 + (bull_market_duration / 30.0), 3.0)  # Scale up to 3x over 30 days
        
        base_params = {
            'stop_loss_pct': -0.18 * duration_multiplier,      # Wide stops (up to -54%)
            'take_profit_pct': 2.0 * duration_multiplier,      # Much higher targets (up to 600%) - was 0.40
            'trailing_stop_pct': -0.50 * duration_multiplier,  # MUCH wider trailing stops (up to -150%)
            'position_size_multiplier': 1.2 * duration_multiplier,  # Larger positions (up to 3.6x)
            'threshold_tightness': 0.8 / max(duration_multiplier, 1.0),  # Tighter thresholds
            'momentum_bias': 0.3 * duration_multiplier,         # Moderate bias toward staying invested
            'trend_following': True,      # Enable trend-following mode
            'conviction_multiplier': 1.5 * duration_multiplier, # Higher conviction (up to 4.5x)
            'min_hold_days': max(7, int(21 * duration_multiplier)),  # Much longer holds (up to 63 days)
            'volatility_protection': False, # DISABLE all volatility protection
            'bull_duration_days': bull_market_duration,
            'duration_scaling': duration_multiplier,
            'overtrading_protection': True,  # Enable overtrading protection
            'buy_and_hold_mode': duration_multiplier > 1.3  # Switch to buy-and-hold mode after 39 days
        }
        
        if bull_market_duration > 20:
            print(f"📈 BULL SCALING: {bull_market_duration} days → {duration_multiplier:.1f}x enhanced aggressiveness")
            
    elif market_regime == "BULL":
        # Moderate bull market duration scaling
        duration_multiplier = min(1.0 + (bull_market_duration / 45.0), 2.5)  # Scale up to 2.5x over 45 days
        
        base_params = {
            'stop_loss_pct': -0.15 * duration_multiplier,      # Wide stops (up to -37.5%)
            'take_profit_pct': 1.5 * duration_multiplier,      # Much higher targets (up to 375%) - was 0.30
            'trailing_stop_pct': -0.40 * duration_multiplier,  # MUCH wider trailing stops (up to -100%)
            'position_size_multiplier': 1.1 * duration_multiplier,  # Larger positions (up to 2.75x)
            'threshold_tightness': 0.9 / max(duration_multiplier, 1.0),  # Tighter thresholds
            'momentum_bias': 0.2 * duration_multiplier,        # Moderate bias
            'trend_following': True,      # Enable trend-following mode
            'conviction_multiplier': 1.3 * duration_multiplier, # Higher conviction (up to 3.25x)
            'min_hold_days': max(5, int(15 * duration_multiplier)),  # Much longer holds (up to 37 days)
            'volatility_protection': False, # DISABLE volatility override in bull markets
            'bull_duration_days': bull_market_duration,
            'duration_scaling': duration_multiplier,
            'overtrading_protection': True,  # Enable overtrading protection
            'buy_and_hold_mode': duration_multiplier > 1.5  # Switch to buy-and-hold mode after 45 days
        }
        
        if bull_market_duration > 30:
            print(f"📈 BULL SCALING: {bull_market_duration} days → {duration_multiplier:.1f}x enhanced aggressiveness")
            
    elif market_regime == "STRONG_BEAR":
        base_params = {
            'stop_loss_pct': -0.04,      # Base stop-loss
            'take_profit_pct': 0.08,     # Base take-profit
            'trailing_stop_pct': -0.05,  # Base trailing stop
            'position_size_multiplier': 0.4,  # Base position size
            'threshold_tightness': 2.2,   # Much wider thresholds (very selective)
            'momentum_bias': -0.3,        # Strong bias toward exiting positions
            'trend_following': False,     # Disable trend-following
            'conviction_multiplier': 0.7, # Lower conviction in bear markets
            'min_hold_days': 0,          # No minimum holding period
            'volatility_protection': False, # No volatility override needed
            'bull_duration_days': 0,     # Reset bull market duration
            'duration_scaling': 1.0
        }
    elif market_regime == "BEAR":
        base_params = {
            'stop_loss_pct': -0.06,      # Base stop-loss
            'take_profit_pct': 0.12,     # Base take-profit
            'trailing_stop_pct': -0.07,  # Base trailing stop
            'position_size_multiplier': 0.6,  # Base position size
            'threshold_tightness': 1.5,   # Wider thresholds (fewer trades)
            'momentum_bias': -0.15,       # Bias toward exiting
            'trend_following': False,     # Disable trend-following
            'conviction_multiplier': 0.8, # Lower conviction
            'min_hold_days': 0,          # No minimum holding period
            'volatility_protection': False, # No volatility override needed
            'bull_duration_days': 0,     # Reset bull market duration
            'duration_scaling': 1.0
        }
    else:  # SIDEWAYS
        base_params = {
            'stop_loss_pct': -0.08,      # Base stop-loss
            'take_profit_pct': 0.25,     # Base take-profit
            'trailing_stop_pct': -0.10,  # Base trailing stop
            'position_size_multiplier': 1.0,  # Base position size
            'threshold_tightness': 1.0,
            'momentum_bias': 0.0,         # No bias
            'trend_following': False,     # Standard mean-reversion
            'conviction_multiplier': 1.0, # Standard conviction
            'min_hold_days': 1,          # Minimum holding period to reduce noise
            'volatility_protection': False, # No volatility override needed
            'bull_duration_days': 0,     # Reset bull market duration
            'duration_scaling': 1.0
        }
    
    # APPLY VOLATILITY ADJUSTMENTS
    adjusted_params = base_params.copy()
    
    # Adjust stop-loss (make wider for high volatility and value strategy)
    adjusted_params['stop_loss_pct'] = base_params['stop_loss_pct'] * vol_stop_multiplier * value_stop_multiplier
    adjusted_params['stop_loss_pct'] = max(adjusted_params['stop_loss_pct'], -0.50)  # Cap at -50%
    
    # Adjust take-profit (make lower for high volatility, but consider value strategy)
    adjusted_params['take_profit_pct'] = base_params['take_profit_pct'] * vol_profit_multiplier * value_profit_multiplier
    adjusted_params['take_profit_pct'] = max(adjusted_params['take_profit_pct'], 0.05)  # Minimum 5%
    
    # Adjust trailing stop (make wider for high volatility and value strategy)
    adjusted_params['trailing_stop_pct'] = base_params['trailing_stop_pct'] * vol_trail_multiplier * value_stop_multiplier
    adjusted_params['trailing_stop_pct'] = max(adjusted_params['trailing_stop_pct'], -0.60)  # Cap at -60%
    
    # Adjust position size (smaller for high volatility, but larger for value strategy)
    adjusted_params['position_size_multiplier'] = base_params['position_size_multiplier'] * vol_position_multiplier * value_position_multiplier
    adjusted_params['position_size_multiplier'] = max(adjusted_params['position_size_multiplier'], 0.2)  # Minimum 20%
    
    # Adjust threshold tightness (value strategy gets wider thresholds for less frequent trading)
    adjusted_params['threshold_tightness'] = base_params['threshold_tightness'] * value_threshold_multiplier
    
    # Adjust minimum holding period (value strategy gets much longer holds)
    adjusted_params['min_hold_days'] = int(base_params['min_hold_days'] * value_min_hold_multiplier)
    
    # Add volatility and strategy metadata
    adjusted_params['price_volatility'] = price_volatility
    adjusted_params['strategy'] = strategy
    adjusted_params['volatility_adjustment'] = {
        'stop_multiplier': vol_stop_multiplier,
        'profit_multiplier': vol_profit_multiplier,
        'position_multiplier': vol_position_multiplier,
        'trail_multiplier': vol_trail_multiplier
    }
    adjusted_params['value_adjustment'] = {
        'stop_multiplier': value_stop_multiplier,
        'profit_multiplier': value_profit_multiplier,
        'position_multiplier': value_position_multiplier,
        'min_hold_multiplier': value_min_hold_multiplier,
        'threshold_multiplier': value_threshold_multiplier
    }
    
    return adjusted_params


def calculate_adaptive_thresholds(score_history: List[float], market_regime: str, lookback: int = 20, strategy: str = "momentum", bull_market_duration: int = 0) -> Tuple[float, float]:
    """
    Calculate adaptive buy/sell thresholds based on market regime, score volatility, strategy, and bull market duration
    Enhanced with volatility override protection, value strategy optimizations, and bull market duration scaling
    
    Args:
        score_history: List of recent Final_Buy_Scores
        market_regime: Current market regime
        lookback: Number of recent scores to analyze
        strategy: Trading strategy ("momentum" or "value") for strategy-specific optimizations
        bull_market_duration: Days in current bull market for adaptive scaling
        
    Returns:
        Tuple of (buy_threshold, sell_threshold)
    """
    if len(score_history) < lookback:
        # Not enough history, use regime-based defaults with strategy-specific adjustments
        if strategy.lower() == "value":
            # VALUE STRATEGY: More conservative thresholds, less frequent trading
            if market_regime == "STRONG_BULL":
                return 5.05, 4.85  # More conservative than momentum
            elif market_regime == "BULL":
                return 5.10, 4.90  # More conservative than momentum
            elif market_regime == "STRONG_BEAR":
                return 5.50, 4.50  # Much wider thresholds
            elif market_regime == "BEAR":
                return 5.35, 4.65  # Wider thresholds
            else:
                return 5.15, 4.85  # More conservative SIDEWAYS
        else:
            # MOMENTUM STRATEGY: Aggressive thresholds for maximum participation
            if market_regime == "STRONG_BULL":
                return 4.85, 4.70  # Aggressive thresholds for maximum participation
            elif market_regime == "BULL":
                return 4.90, 4.80  # Aggressive thresholds
            elif market_regime == "STRONG_BEAR":
                return 5.35, 4.65  # Very wide thresholds for strong bear markets
            elif market_regime == "BEAR":
                return 5.20, 4.80  # Wider thresholds for selectivity
            else:
                return 5.02, 4.98  # Standard thresholds
    
    recent_scores = score_history[-lookback:]
    mean_score = np.mean(recent_scores)
    std_score = np.std(recent_scores)
    
    # Get regime-specific parameters with strategy and bull market duration
    risk_params = get_adaptive_risk_params(market_regime, strategy=strategy, bull_market_duration=bull_market_duration)
    tightness = risk_params['threshold_tightness']
    momentum_bias = risk_params.get('momentum_bias', 0.0)
    
    # BULL MARKET DURATION SCALING: Make thresholds more aggressive as bull market continues
    duration_scaling = 1.0
    if market_regime in ["BULL", "STRONG_BULL"] and bull_market_duration > 20:
        # Moderate scaling for sustained bull markets
        if market_regime == "STRONG_BULL":
            duration_scaling = min(1.0 + (bull_market_duration / 30.0), 3.0)  # Up to 3x more aggressive over 30 days
        else:  # BULL
            duration_scaling = min(1.0 + (bull_market_duration / 45.0), 2.5)  # Up to 2.5x more aggressive over 45 days
        
        tightness /= duration_scaling  # Tighter thresholds
        momentum_bias *= duration_scaling  # Stronger momentum bias
        
        if bull_market_duration > 30:
            print(f"📈 THRESHOLD SCALING: {bull_market_duration} days → {duration_scaling:.1f}x threshold tightening")
    
    # VALUE STRATEGY ADJUSTMENTS: Wider thresholds for less frequent trading
    if strategy.lower() == "value":
        # Value strategy benefits from wider thresholds to avoid overtrading
        tightness *= 1.5  # Make thresholds 50% wider
        momentum_bias *= 0.5  # Reduce momentum bias for value stocks
        
        # Only print once per function call
        if not hasattr(calculate_adaptive_thresholds, '_value_msg_shown'):
            print(f"📊 VALUE THRESHOLD ADJUSTMENT: Making thresholds 50% wider for less frequent trading")
            calculate_adaptive_thresholds._value_msg_shown = True
    
    # VOLATILITY OVERRIDE: Detect extreme volatility scenarios
    volatility_threshold = 0.3  # If std_score > 0.3, consider it extreme volatility
    is_extreme_volatility = std_score > volatility_threshold
    
    # Adaptive volatility factor with volatility override protection
    base_volatility_factor = min(std_score * 2, 0.8)  # Increased cap from 0.5 to 0.8
    
    # Apply volatility override - force wider thresholds for extreme volatility
    if is_extreme_volatility:
        # Override aggressive regimes with conservative parameters
        if market_regime in ["STRONG_BULL", "BULL"]:
            print(f"⚠️  VOLATILITY OVERRIDE: Extreme volatility detected (σ={std_score:.3f}), applying conservative thresholds despite {market_regime} regime")
            tightness = max(tightness, 1.8)  # Force wider thresholds
            momentum_bias = min(momentum_bias, 0.1)  # Reduce momentum bias
        
        # Extra volatility protection
        volatility_protection_factor = min(std_score * 1.5, 1.0)  # Additional widening
        adjusted_volatility_factor = (base_volatility_factor + volatility_protection_factor) * tightness
    else:
        # Normal volatility adjustment
        adjusted_volatility_factor = base_volatility_factor * tightness
    
    # Enhanced momentum bias for bull markets (reduced for value strategy)
    bias_adjustment = momentum_bias * 0.15
    
    # Calculate thresholds with volatility protection
    buy_threshold = mean_score + adjusted_volatility_factor - bias_adjustment
    sell_threshold = mean_score - adjusted_volatility_factor - bias_adjustment
    
    # Apply regime-specific bounds with volatility override protection and strategy adjustments
    if is_extreme_volatility:
        # Use conservative bounds regardless of regime
        buy_threshold = max(5.05, min(buy_threshold, 6.0))   # Conservative bounds for extreme volatility
        sell_threshold = max(4.0, min(sell_threshold, 4.95))
    else:
        # Normal regime-specific bounds with strategy adjustments and ULTRA-AGGRESSIVE bull market bounds
        if strategy.lower() == "value":
            # VALUE STRATEGY: More conservative bounds across all regimes
            if market_regime == "STRONG_BULL":
                buy_threshold = max(4.85, min(buy_threshold, 5.2))   # More conservative than momentum
                sell_threshold = max(4.3, min(sell_threshold, 4.80))
            elif market_regime == "BULL":
                buy_threshold = max(4.95, min(buy_threshold, 5.3))   # More conservative than momentum
                sell_threshold = max(4.4, min(sell_threshold, 4.90))
            elif market_regime == "STRONG_BEAR":
                buy_threshold = max(5.50, min(buy_threshold, 6.8))   # Much more conservative
                sell_threshold = max(3.2, min(sell_threshold, 4.50))
            elif market_regime == "BEAR":
                buy_threshold = max(5.30, min(buy_threshold, 6.3))   # More conservative
                sell_threshold = max(3.7, min(sell_threshold, 4.70))
            else:  # SIDEWAYS
                buy_threshold = max(5.15, min(buy_threshold, 6.0))   # More conservative
                sell_threshold = max(4.0, min(sell_threshold, 4.85))
        else:
            # MOMENTUM STRATEGY: Moderate bounds for bull market participation
            if market_regime == "STRONG_BULL":
                # Aggressive: Buy more, sell less
                buy_threshold = max(4.70, min(buy_threshold, 5.0))   # Aggressive bounds
                sell_threshold = max(4.0, min(sell_threshold, 4.70))  # Reluctant to sell
            elif market_regime == "BULL":
                # Moderately aggressive: Bias toward buying and holding
                buy_threshold = max(4.80, min(buy_threshold, 5.1))   # Moderately aggressive bounds
                sell_threshold = max(4.2, min(sell_threshold, 4.80))  # Moderately reluctant to sell
            elif market_regime == "STRONG_BEAR":
                buy_threshold = max(5.30, min(buy_threshold, 6.5))   # Very conservative bounds
                sell_threshold = max(3.5, min(sell_threshold, 4.70))
            elif market_regime == "BEAR":
                buy_threshold = max(5.10, min(buy_threshold, 6.0))   # Conservative bounds
                sell_threshold = max(4.0, min(sell_threshold, 4.90))
            else:  # SIDEWAYS
                buy_threshold = max(5.01, min(buy_threshold, 5.8))
                sell_threshold = max(4.2, min(sell_threshold, 4.99))
    
    return buy_threshold, sell_threshold


def calculate_conviction_score(final_buy_score: float, score_history: List[float], market_regime: str) -> float:
    """
    Calculate conviction score based on signal strength and consistency
    Enhanced with volatility protection to prevent overexposure in volatile conditions
    
    Args:
        final_buy_score: Current Final_Buy_Score
        score_history: Historical scores for trend analysis
        market_regime: Current market regime
        
    Returns:
        Conviction score (0.3 to 2.0) - multiplier for position sizing with volatility protection
    """
    base_conviction = 1.0
    
    # Distance from neutral (5.0) indicates signal strength
    signal_strength = abs(final_buy_score - 5.0)
    conviction_boost = min(signal_strength * 0.3, 0.5)  # Max 0.5 boost
    
    # Trend consistency bonus with volatility penalty
    if len(score_history) >= 5:
        recent_scores = score_history[-5:]
        score_volatility = np.std(recent_scores)
        trend_consistency = 1.0 - (score_volatility / 2.0)  # Lower volatility = higher consistency
        trend_consistency = max(0, min(trend_consistency, 0.3))  # Max 0.3 boost
        conviction_boost += trend_consistency
        
        # VOLATILITY PROTECTION: Reduce conviction for extreme score volatility
        if score_volatility > 0.3:  # Extreme volatility threshold
            volatility_penalty = min(score_volatility * 0.5, 0.4)  # Max 0.4 penalty
            conviction_boost -= volatility_penalty
            print(f"⚠️  VOLATILITY PROTECTION: High score volatility (σ={score_volatility:.3f}), reducing position size")
    
    # Market regime bonus with volatility override
    risk_params = get_adaptive_risk_params(market_regime)
    volatility_protection = risk_params.get('volatility_protection', False)
    
    if market_regime in ["STRONG_BULL", "BULL"]:
        regime_bonus = 0.2  # Extra conviction in bull markets
        
        # Apply volatility protection for bull markets
        if volatility_protection and len(score_history) >= 10:
            longer_volatility = np.std(score_history[-10:])
            if longer_volatility > 0.25:  # High volatility over longer period
                regime_bonus *= 0.5  # Reduce bull market bonus
                print(f"⚠️  VOLATILITY OVERRIDE: Reducing bull market conviction due to high volatility (σ={longer_volatility:.3f})")
        
        conviction_boost += regime_bonus
    elif market_regime in ["STRONG_BEAR"]:
        conviction_boost -= 0.2  # Reduced conviction in strong bear markets
    
    final_conviction = base_conviction + conviction_boost
    return max(0.3, min(final_conviction, 2.0))  # Clamp between 0.3x and 2.0x with lower minimum for volatility protection


def detect_momentum_reversal(price_data: pd.DataFrame, score_history: List[float], lookback_days: int = 10) -> Dict:
    """
    Detect momentum reversals using both price and sentiment signals
    
    Args:
        price_data: Historical price data
        score_history: Recent Final_Buy_Scores
        lookback_days: Days to analyze for reversal detection
        
    Returns:
        Dictionary with reversal signals and strength
    """
    try:
        if len(price_data) < lookback_days or len(score_history) < 5:
            return {'reversal_detected': False, 'type': None, 'strength': 0.0}
        
        recent_prices = price_data.tail(lookback_days)['Close']
        recent_scores = score_history[-lookback_days:] if len(score_history) >= lookback_days else score_history
        
        # PRICE MOMENTUM SIGNALS
        # 1. Price acceleration (second derivative)
        price_returns = recent_prices.pct_change().dropna()
        if len(price_returns) >= 3:
            recent_acceleration = price_returns.iloc[-3:].mean() - price_returns.iloc[-6:-3].mean()
        else:
            recent_acceleration = 0
            
        # 2. Price trend strength
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        # 3. Price volatility (high volatility can signal reversals)
        price_volatility = price_returns.std() if len(price_returns) > 1 else 0
        
        # SENTIMENT MOMENTUM SIGNALS
        # 1. Sentiment acceleration
        if len(recent_scores) >= 6:
            sentiment_acceleration = np.mean(recent_scores[-3:]) - np.mean(recent_scores[-6:-3])
        else:
            sentiment_acceleration = 0
            
        # 2. Sentiment trend
        if len(recent_scores) >= 2:
            sentiment_trend = recent_scores[-1] - recent_scores[0]
        else:
            sentiment_trend = 0
            
        # 3. Sentiment-Price divergence (key reversal signal)
        sentiment_direction = 1 if sentiment_trend > 0 else -1
        price_direction = 1 if price_trend > 0 else -1
        divergence = sentiment_direction != price_direction and abs(sentiment_trend) > 0.1 and abs(price_trend) > 0.02
        
        # REVERSAL DETECTION LOGIC
        reversal_signals = []
        
        # Bullish reversal signals
        if (recent_acceleration > 0.01 and sentiment_acceleration > 0.1) or \
           (price_trend > 0.05 and sentiment_trend > 0.2) or \
           (divergence and sentiment_direction > 0):
            reversal_signals.append(('BULLISH', 0.7 if divergence else 0.5))
            
        # Bearish reversal signals  
        if (recent_acceleration < -0.01 and sentiment_acceleration < -0.1) or \
           (price_trend < -0.05 and sentiment_trend < -0.2) or \
           (divergence and sentiment_direction < 0):
            reversal_signals.append(('BEARISH', 0.7 if divergence else 0.5))
            
        # Strong momentum continuation (not reversal, but important)
        if abs(recent_acceleration) > 0.02 and abs(sentiment_acceleration) > 0.15 and not divergence:
            direction = 'BULLISH' if recent_acceleration > 0 else 'BEARISH'
            reversal_signals.append((f'{direction}_ACCELERATION', 0.8))
        
        # Determine strongest signal
        if reversal_signals:
            strongest_signal = max(reversal_signals, key=lambda x: x[1])
            return {
                'reversal_detected': True,
                'type': strongest_signal[0],
                'strength': strongest_signal[1],
                'price_trend': price_trend,
                'sentiment_trend': sentiment_trend,
                'divergence': divergence,
                'acceleration': recent_acceleration,
                'sentiment_acceleration': sentiment_acceleration
            }
        else:
            return {
                'reversal_detected': False,
                'type': None,
                'strength': 0.0,
                'price_trend': price_trend,
                'sentiment_trend': sentiment_trend,
                'divergence': divergence
            }
            
    except Exception as e:
        print(f"⚠️  Momentum reversal detection failed: {e}")
        return {'reversal_detected': False, 'type': None, 'strength': 0.0}


def get_trading_recommendation(ticker: str, final_buy_score: float, sentiment_df: pd.DataFrame = None, price_data: pd.DataFrame = None, strategy: str = "momentum") -> Dict:
    """
    Convert Final_Buy_Score into concrete BUY/HOLD/SELL recommendation
    Uses adaptive thresholds, market regime detection, volatility-adjusted risk management, momentum reversal detection, and strategy-specific optimizations
    
    Args:
        ticker: Stock symbol
        final_buy_score: Current Final_Buy_Score
        sentiment_df: Historical sentiment data for score history (optional)
        price_data: Historical price data for volatility calculation (optional)
        strategy: Trading strategy ("momentum" or "value") for strategy-specific optimizations
        
    Returns:
        Dictionary with trading recommendation and details
    """
    # Calculate price volatility for risk adjustment
    price_volatility = calculate_price_volatility(ticker, price_data)
    
    # Detect current market regime
    market_regime = detect_market_regime(ticker, price_data)
    
    # Calculate bull market duration for adaptive scaling
    bull_market_duration = calculate_bull_market_duration(price_data, market_regime) if price_data is not None else 0
    
    # Get historical Final_Buy_Scores if available
    score_history = []
    if sentiment_df is not None and hasattr(sentiment_df, 'attrs') and 'Final_Buy_Scores_Over_Time' in sentiment_df.attrs:
        scores_over_time = sentiment_df.attrs['Final_Buy_Scores_Over_Time']
        score_history = [score for _, score in scores_over_time]
    else:
        # Fallback: use current score repeated (not ideal but functional)
        score_history = [final_buy_score] * 10
    
    # MOMENTUM REVERSAL DETECTION: Check for momentum reversals that could override regime
    reversal_info = {'reversal_detected': False, 'type': None, 'strength': 0.0}
    effective_market_regime = market_regime
    
    if price_data is not None and len(score_history) >= 5:
        reversal_info = detect_momentum_reversal(price_data, score_history)
        
        if reversal_info['reversal_detected'] and reversal_info['strength'] > 0.6:
            # Strong reversal detected - override market regime
            reversal_type = reversal_info['type']
            strength = reversal_info['strength']
            
            if 'BULLISH' in reversal_type:
                if market_regime in ["BEAR", "STRONG_BEAR", "SIDEWAYS"]:
                    if 'ACCELERATION' in reversal_type:
                        effective_market_regime = "STRONG_BULL"
                        print(f"🔄 MOMENTUM REVERSAL: Strong bullish acceleration detected (strength: {strength:.1f}) - overriding {market_regime} → {effective_market_regime}")
                    else:
                        effective_market_regime = "BULL"
                        print(f"🔄 MOMENTUM REVERSAL: Bullish reversal detected (strength: {strength:.1f}) - overriding {market_regime} → {effective_market_regime}")
                elif market_regime == "BULL" and 'ACCELERATION' in reversal_type:
                    effective_market_regime = "STRONG_BULL"
                    print(f"� MOMENTUM REVERSAL: Bullish acceleration detected (strength: {strength:.1f}) - upgrading {market_regime} → {effective_market_regime}")
                    
            elif 'BEARISH' in reversal_type:
                if market_regime in ["BULL", "STRONG_BULL", "SIDEWAYS"]:
                    if 'ACCELERATION' in reversal_type:
                        effective_market_regime = "STRONG_BEAR"
                        print(f"🔄 MOMENTUM REVERSAL: Strong bearish acceleration detected (strength: {strength:.1f}) - overriding {market_regime} → {effective_market_regime}")
                    else:
                        effective_market_regime = "BEAR"
                        print(f"🔄 MOMENTUM REVERSAL: Bearish reversal detected (strength: {strength:.1f}) - overriding {market_regime} → {effective_market_regime}")
                elif market_regime == "BEAR" and 'ACCELERATION' in reversal_type:
                    effective_market_regime = "STRONG_BEAR"
                    print(f"🔄 MOMENTUM REVERSAL: Bearish acceleration detected (strength: {strength:.1f}) - upgrading {market_regime} → {effective_market_regime}")
    
    # LEGACY MOMENTUM OVERRIDE: Upgrade market regime if strong momentum detected (kept for backward compatibility)
    if len(score_history) >= 5 and not reversal_info['reversal_detected']:
        recent_score_trend = np.mean(score_history[-3:]) - np.mean(score_history[-6:-3]) if len(score_history) >= 6 else 0
        if recent_score_trend > 0.15 and final_buy_score > 5.1:
            if market_regime == "SIDEWAYS":
                effective_market_regime = "BULL"
                print(f"🚀 MOMENTUM OVERRIDE: Upgrading {market_regime} → {effective_market_regime} (trend: +{recent_score_trend:.2f})")
            elif market_regime == "BULL":
                effective_market_regime = "STRONG_BULL"
                print(f"🚀 MOMENTUM OVERRIDE: Upgrading {market_regime} → {effective_market_regime} (trend: +{recent_score_trend:.2f})")
    
    # Calculate adaptive thresholds with effective regime and strategy
    buy_threshold, sell_threshold = calculate_adaptive_thresholds(score_history, effective_market_regime, strategy=strategy)
    
    # Get volatility-adjusted risk parameters with effective regime, strategy, and bull market duration
    risk_params = get_adaptive_risk_params(effective_market_regime, price_volatility, strategy=strategy, bull_market_duration=bull_market_duration)
    
    # Enhanced decision logic with ultra-aggressive bull market optimizations
    trend_following = risk_params.get('trend_following', False)
    momentum_bias = risk_params.get('momentum_bias', 0.0)
    conviction_multiplier = risk_params.get('conviction_multiplier', 1.0)
    
    # Calculate conviction score for dynamic position sizing
    conviction_score = calculate_conviction_score(final_buy_score, score_history, effective_market_regime)
    
    # Apply enhanced trend-following logic for bull markets with moderate scaling
    if trend_following and effective_market_regime in ["BULL", "STRONG_BULL"]:
        # In bull markets, be more aggressive about buying and holding
        if effective_market_regime == "STRONG_BULL":
            effective_buy_threshold = buy_threshold * 0.85  # 15% more aggressive
            effective_sell_threshold = sell_threshold * 0.70  # 30% less likely to sell
        else:  # BULL
            effective_buy_threshold = buy_threshold * 0.90  # 10% more aggressive
            effective_sell_threshold = sell_threshold * 0.80  # 20% less likely to sell
        
        # Moderate momentum persistence - if score is trending up, be more aggressive
        if len(score_history) >= 3:
            recent_trend = np.mean(score_history[-3:]) - np.mean(score_history[-6:-3]) if len(score_history) >= 6 else 0
            if recent_trend > 0.01:  # Moderate threshold for positive momentum
                if effective_market_regime == "STRONG_BULL":
                    effective_buy_threshold *= 0.90  # More aggressive
                    effective_sell_threshold *= 0.80  # Less likely to sell
                else:
                    effective_buy_threshold *= 0.95  # Slightly more aggressive
                    effective_sell_threshold *= 0.85  # Slightly less likely to sell
                    
        # Momentum lock: If we've been in bull market for a while, lock in momentum
        if effective_market_regime == "STRONG_BULL" and bull_market_duration > 60:
            effective_buy_threshold = min(effective_buy_threshold, 4.5)  # Force buying at 4.5
            effective_sell_threshold = min(effective_sell_threshold, 4.0)  # Reluctant to sell above 4.0
        elif effective_market_regime == "BULL" and bull_market_duration > 90:
            effective_buy_threshold = min(effective_buy_threshold, 4.7)  # Force buying at 4.7
            effective_sell_threshold = min(effective_sell_threshold, 4.2)  # Reluctant to sell above 4.2
                    
        # Momentum reversal protection - if strong downward trend, be more cautious
        if len(score_history) >= 5:
            recent_decline = score_history[-1] - score_history[-5]
            if recent_decline < -0.3:  # Moderate threshold for decline
                effective_sell_threshold *= 1.15  # More likely to sell
    else:
        effective_buy_threshold = buy_threshold
        effective_sell_threshold = sell_threshold
    
    # MOMENTUM REVERSAL ADJUSTMENTS: Fine-tune thresholds based on reversal signals
    if reversal_info['reversal_detected']:
        reversal_type = reversal_info['type']
        strength = reversal_info['strength']
        
        if 'BULLISH' in reversal_type:
            # Make buying more aggressive during bullish reversals
            threshold_adjustment = 1.0 - (strength * 0.1)  # Up to 10% more aggressive
            effective_buy_threshold *= threshold_adjustment
            print(f"🎯 REVERSAL ADJUSTMENT: Making buy threshold {(1-threshold_adjustment)*100:.1f}% more aggressive due to {reversal_type}")
            
        elif 'BEARISH' in reversal_type:
            # Make selling more aggressive during bearish reversals
            threshold_adjustment = 1.0 + (strength * 0.1)  # Up to 10% more likely to sell
            effective_sell_threshold *= threshold_adjustment
            print(f"🎯 REVERSAL ADJUSTMENT: Making sell threshold {(threshold_adjustment-1)*100:.1f}% more aggressive due to {reversal_type}")
    
    # Determine recommendation with ULTRA-AGGRESSIVE logic and conviction scoring
    if final_buy_score >= effective_buy_threshold:
        recommendation = "BUY"
        confidence = min(100, ((final_buy_score - effective_buy_threshold) / (10 - effective_buy_threshold)) * 100)
        
        # Moderate confidence boost in bull markets with conviction
        if effective_market_regime == "STRONG_BULL":
            confidence = min(100, confidence * 3.0 * conviction_score)  # Higher confidence
        elif effective_market_regime == "BULL":
            confidence = min(100, confidence * 2.5 * conviction_score)  # Higher confidence
            
        reasoning = f"Score {final_buy_score:.2f} exceeds {effective_market_regime} buy threshold {effective_buy_threshold:.3f}"
        
        if trend_following:
            reasoning += f" (ultra-aggressive trend-following, conviction: {conviction_score:.1f}x)"
        if reversal_info['reversal_detected']:
            reasoning += f" (momentum reversal: {reversal_info['type']}, strength: {reversal_info['strength']:.1f})"
            
    elif final_buy_score <= effective_sell_threshold:
        recommendation = "SELL"
        confidence = min(100, ((effective_sell_threshold - final_buy_score) / effective_sell_threshold) * 100)
        
        # Apply conviction to sell signals too, but reduce in bull markets
        if effective_market_regime in ["BULL", "STRONG_BULL"]:
            confidence = min(100, confidence * conviction_score * 0.5)  # Reduce sell confidence in bull markets
        else:
            confidence = min(100, confidence * conviction_score)
        
        reasoning = f"Score {final_buy_score:.2f} below {effective_market_regime} sell threshold {effective_sell_threshold:.3f}"
        
        if reversal_info['reversal_detected']:
            reasoning += f" (momentum reversal: {reversal_info['type']}, strength: {reversal_info['strength']:.1f})"
    else:
        recommendation = "HOLD"
        confidence = 100 - abs(final_buy_score - 5.0) * 20  # Closer to neutral = higher confidence
        
        # In bull markets, moderate bias toward HOLD instead of SELL
        if effective_market_regime in ["BULL", "STRONG_BULL"] and final_buy_score > 4.0:
            if effective_market_regime == "STRONG_BULL":
                confidence = min(100, confidence * 2.5 * conviction_score)  # Higher confidence in holding
            else:  # BULL
                confidence = min(100, confidence * 2.0 * conviction_score)  # Higher confidence in holding
            
        reasoning = f"Score {final_buy_score:.2f} within {effective_market_regime} hold range ({effective_sell_threshold:.3f} - {effective_buy_threshold:.3f})"
        
        if reversal_info['reversal_detected']:
            reasoning += f" (monitoring momentum reversal: {reversal_info['type']})"
    
    return {
        'recommendation': recommendation,
        'confidence': max(0, min(100, confidence)),
        'market_regime': effective_market_regime,  # Use effective regime
        'original_regime': market_regime,  # Keep original for reference
        'buy_threshold': effective_buy_threshold,
        'sell_threshold': effective_sell_threshold,
        'reasoning': reasoning,
        'risk_params': risk_params,
        'final_buy_score': final_buy_score,
        'trend_following': trend_following,
        'momentum_bias': momentum_bias,
        'conviction_score': conviction_score,
        'position_size_multiplier': risk_params['position_size_multiplier'] * conviction_score,
        'momentum_reversal': reversal_info  # Include reversal information
    }