"""
Shared Logic Module
Contains market regime detection, adaptive thresholds, and risk management logic
Used by both main program and backtester
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
        print(f"Bull market duration calculation failed: {e}")
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
    if market_regime not in ["BULL", "STRONG_BULL"] or entry_price <= 0:
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


def detect_choppy_market(prices: pd.Series, lookback_days: int = 30) -> dict:
    """
    Detect choppy/sideways market conditions that are unfavorable for trading
    
    Args:
        prices: Series of closing prices
        lookback_days: Number of days to analyze
        
    Returns:
        Dict with choppy market analysis
    """
    try:
        if len(prices) < max(10, lookback_days // 2):
            return {'is_choppy': False, 'choppiness_score': 0.0, 'reason': 'Insufficient data'}
        
        recent_prices = prices.tail(lookback_days)
        returns = recent_prices.pct_change().dropna()
        
        if len(returns) < 5:
            return {'is_choppy': False, 'choppiness_score': 0.0, 'reason': 'Insufficient returns data'}
        
        # 1. DIRECTIONAL MOVEMENT INDEX (ADX-like calculation)
        # Measures trend strength - low values indicate choppy markets
        high_low_range = recent_prices.rolling(2).apply(lambda x: abs(x.iloc[1] - x.iloc[0]), raw=False)
        avg_range = high_low_range.mean()
        
        # Calculate directional movement
        up_moves = returns[returns > 0].sum()
        down_moves = abs(returns[returns < 0].sum())
        total_movement = up_moves + down_moves
        
        if total_movement > 0:
            directional_index = abs(up_moves - down_moves) / total_movement
        else:
            directional_index = 0
        
        # 2. PRICE OSCILLATION ANALYSIS
        # Count how many times price crosses the moving average (whipsaws)
        sma_10 = recent_prices.rolling(10).mean()
        price_above_sma = recent_prices > sma_10
        crossovers = (price_above_sma != price_above_sma.shift(1)).sum()
        crossover_rate = crossovers / len(recent_prices) if len(recent_prices) > 0 else 0
        
        # 3. VOLATILITY vs TREND ANALYSIS
        # High volatility with low net movement indicates choppiness
        volatility = returns.std() * (252 ** 0.5)  # Annualized
        net_return = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        if volatility > 0:
            efficiency_ratio = abs(net_return) / volatility
        else:
            efficiency_ratio = 0
        
        # 4. CONSECUTIVE REVERSALS
        # Count consecutive days of direction changes
        direction_changes = (returns.shift(1) * returns < 0).sum()
        reversal_rate = direction_changes / len(returns) if len(returns) > 0 else 0
        
        # 5. RANGE-BOUND BEHAVIOR
        # Check if price is stuck in a narrow range
        price_range = (recent_prices.max() - recent_prices.min()) / recent_prices.mean()
        
        # CHOPPINESS SCORING (0-1, higher = more choppy)
        choppiness_factors = {
            'low_directional_movement': max(0, 0.3 - directional_index) * 3.33,  # 0-1
            'high_crossover_rate': min(1, crossover_rate * 5),  # 0-1
            'low_efficiency': max(0, 0.1 - efficiency_ratio) * 10,  # 0-1
            'high_reversal_rate': min(1, reversal_rate * 2),  # 0-1
            'narrow_range': max(0, 0.15 - price_range) * 6.67 if price_range < 0.15 else 0  # 0-1
        }
        
        # Weighted choppiness score
        weights = {
            'low_directional_movement': 0.25,
            'high_crossover_rate': 0.25,
            'low_efficiency': 0.20,
            'high_reversal_rate': 0.20,
            'narrow_range': 0.10
        }
        
        choppiness_score = sum(choppiness_factors[factor] * weights[factor] 
                              for factor in choppiness_factors)
        
        # CHOPPY MARKET THRESHOLDS
        choppy_threshold = 0.6  # Above this = choppy market
        very_choppy_threshold = 0.8  # Above this = very choppy
        
        is_choppy = choppiness_score > choppy_threshold
        is_very_choppy = choppiness_score > very_choppy_threshold
        
        # Generate reason
        main_factors = [factor for factor, score in choppiness_factors.items() 
                       if score > 0.3]
        
        if is_very_choppy:
            reason = f"VERY CHOPPY: {', '.join(main_factors[:2])}"
        elif is_choppy:
            reason = f"CHOPPY: {', '.join(main_factors[:2])}"
        else:
            reason = "TRENDING: Directional movement detected"
        
        return {
            'is_choppy': is_choppy,
            'is_very_choppy': is_very_choppy,
            'choppiness_score': choppiness_score,
            'directional_index': directional_index,
            'crossover_rate': crossover_rate,
            'efficiency_ratio': efficiency_ratio,
            'reversal_rate': reversal_rate,
            'price_range': price_range,
            'factors': choppiness_factors,
            'reason': reason
        }
        
    except Exception as e:
        return {'is_choppy': False, 'choppiness_score': 0.0, 'reason': f'Error: {e}'}


def detect_market_regime(ticker: str = None, price_data: pd.DataFrame = None, lookback_days: int = 30) -> str:
    """
    Detect current market regime (BULL, BEAR, SIDEWAYS) for adaptive strategy
    REFINED: Choppiness override for early bull transitions, softened return thresholds, exposure preservation
    
    Args:
        ticker: Stock symbol (if price_data not provided)
        price_data: Historical price data (if ticker not provided)
        lookback_days: Days to analyze for regime detection
        
    Returns:
        Market regime: "BULL", "STRONG_BULL", "BEAR", "STRONG_BEAR", "SIDEWAYS", "CHOPPY", or "CHOPPY_SIDEWAYS"
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
        
        if len(prices) < 2:
            return "SIDEWAYS"  # Need at least 2 data points
        
        # Calculate trend metrics
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        
        if start_price <= 0:
            return "SIDEWAYS"  # Invalid start price
            
        total_return = (end_price - start_price) / start_price
        
        # Calculate moving averages for trend confirmation
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else sma_20
        
        # Calculate volatility (standard deviation of returns)
        returns = prices.pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
        
        # Calculate momentum indicators
        if len(prices) >= 20:
            recent_10_mean = prices.iloc[-10:].mean()
            prev_10_mean = prices.iloc[-20:-10].mean()
            if prev_10_mean > 0:
                recent_momentum = (recent_10_mean - prev_10_mean) / prev_10_mean
            else:
                recent_momentum = 0
        else:
            recent_momentum = 0
        
        # MEDIUM-TERM MOMENTUM for choppiness override (20-30 day window)
        medium_term_return = 0
        if len(prices) >= 25:
            medium_start = prices.iloc[-25]
            medium_end = prices.iloc[-1]
            if medium_start > 0:
                medium_term_return = (medium_end - medium_start) / medium_start
        
        # TREND PERSISTENCE: Check if price is above medium-term moving average
        trend_persistence = end_price > sma_20
        
        # CHOPPY MARKET DETECTION - Critical for avoiding whipsaw losses
        choppy_analysis = detect_choppy_market(prices, lookback_days)
        choppiness_score = choppy_analysis['choppiness_score']
        is_choppy = choppy_analysis['is_choppy']
        is_very_choppy = choppy_analysis['is_very_choppy']
        
        # CHOPPINESS OVERRIDE (Critical): Allow bull behavior when trend and momentum align
        # Prevents delayed re-entry during early bull transitions
        choppy_override = False
        if (is_choppy or is_very_choppy):
            # Check if medium-term momentum is positive AND trend persistence exists
            if medium_term_return > 0.05 and trend_persistence and recent_momentum > 0.02:
                choppy_override = True
                print(f"CHOPPINESS OVERRIDE: Medium-term momentum (+{medium_term_return:.1%}) and trend persistence override choppy classification")
        
        # Apply choppiness classification only if override conditions not met
        if is_very_choppy and not choppy_override:
            return "CHOPPY_SIDEWAYS"
        elif is_choppy and not choppy_override:
            return "CHOPPY"
        
        # REDUCED VOLATILITY OVERRIDE: Only apply in extreme cases and not in clear bull trends
        extreme_volatility_threshold = 1.2  # 120% annualized volatility
        
        if volatility > extreme_volatility_threshold:
            # Check if this is a strong uptrend that deserves volatility tolerance
            if len(prices) >= 20:
                recent_20_start = prices.iloc[-20]
                recent_20_end = prices.iloc[-1]
                if recent_20_start > 0:
                    recent_20_return = (recent_20_end - recent_20_start) / recent_20_start
                else:
                    recent_20_return = 0
            else:
                recent_20_return = total_return
            
            # BULL MARKET VOLATILITY TOLERANCE: Allow much higher volatility in bull trends
            if recent_20_return > 0.15 and end_price > sma_20 and recent_momentum > 0.03:
                # Strong recent bull market with momentum - allow very high volatility
                bull_volatility_threshold = 2.0  # Up to 200% volatility allowed in bull markets
                if volatility <= bull_volatility_threshold:
                    # Continue with normal bull market detection below
                    pass
                else:
                    return "SIDEWAYS"  # Only override if volatility is truly extreme (>200%)
            else:
                # Not a clear bull trend - apply volatility override at lower threshold
                if volatility > 1.5:  # 150% threshold for non-bull trends
                    return "SIDEWAYS"
        
        # SOFTENED RETURN THRESHOLDS: Complement absolute returns with trend persistence and momentum
        # Prevents suppression of slow bull markets during low-volatility grind-ups
        
        # Strong bull: High returns OR strong trend persistence with momentum
        if (total_return > 0.20 and end_price > sma_20 > sma_50 and recent_momentum > 0.05) or \
           (total_return > 0.12 and trend_persistence and recent_momentum > 0.04 and medium_term_return > 0.08):
            return "STRONG_BULL"
        
        # Regular bull: Moderate returns OR trend persistence with positive momentum
        elif (total_return > 0.08 and end_price > sma_20 and recent_momentum > 0.025) or \
             (total_return > 0.04 and trend_persistence and recent_momentum > 0.02 and medium_term_return > 0.04):
            return "BULL"
        
        # Mild bull: Favor participation - low returns but clear trend and momentum
        elif total_return > 0.02 and trend_persistence and recent_momentum > 0.015:
            return "BULL"  # Gradual bullish classification during grind-ups
        
        # Bear markets: Require stronger evidence for bear classification (asymmetric transitions)
        elif total_return < -0.25 and end_price < sma_20 < sma_50 and recent_momentum < -0.05:
            return "STRONG_BEAR"  # Strong downtrend - higher threshold
        elif total_return < -0.15 and end_price < sma_20 and recent_momentum < -0.03:
            return "BEAR"  # Downtrend - higher threshold
        else:
            # EXPOSURE PRESERVATION: When ambiguous, favor maintaining exposure
            # Momentum override for borderline cases - favor bull classification
            if recent_momentum > 0.02 or (trend_persistence and medium_term_return > 0.02):
                return "BULL"  # Default to maintaining exposure when ambiguous
            return "SIDEWAYS"  # Neutral market
            
    except Exception as e:
        print(f"Market regime detection failed: {e}")
        return "SIDEWAYS"  # Safe default


# Global regime state tracking for transition smoothing
_regime_history = []
_current_stable_regime = None

def get_stable_market_regime(ticker: str = None, price_data: pd.DataFrame = None, lookback_days: int = 30) -> str:
    """
    Get market regime with transition smoothing to prevent flip-flopping
    REFINED: Asymmetric transitions, regime persistence requirements, exposure preservation
    
    Args:
        ticker: Stock symbol (if price_data not provided)
        price_data: Historical price data (if ticker not provided)
        lookback_days: Days to analyze for regime detection
        
    Returns:
        Stable market regime with transition smoothing applied
    """
    global _regime_history, _current_stable_regime
    
    # Get raw regime classification
    raw_regime = detect_market_regime(ticker, price_data, lookback_days)
    
    # Initialize if first call
    if _current_stable_regime is None:
        _current_stable_regime = raw_regime
        _regime_history = [raw_regime]
        return raw_regime
    
    # Add to history (keep last 5 evaluations for smoothing)
    _regime_history.append(raw_regime)
    if len(_regime_history) > 5:
        _regime_history.pop(0)
    
    # REGIME TRANSITION SMOOTHING with asymmetric requirements
    consecutive_needed = 2  # Minimum consecutive evaluations for regime change
    
    # Count consecutive occurrences of new regime
    consecutive_count = 0
    for i in range(len(_regime_history) - 1, -1, -1):
        if _regime_history[i] == raw_regime:
            consecutive_count += 1
        else:
            break
    
    # ASYMMETRIC TRANSITION RULES
    current_is_bull = _current_stable_regime in ["BULL", "STRONG_BULL"]
    new_is_bull = raw_regime in ["BULL", "STRONG_BULL"]
    current_is_bear = _current_stable_regime in ["BEAR", "STRONG_BEAR"]
    new_is_bear = raw_regime in ["BEAR", "STRONG_BEAR"]
    
    # EXPOSURE PRESERVATION INVARIANT: Once bullish exposure established, don't reduce without strong evidence
    if current_is_bull and not new_is_bull:
        # Transitioning away from bull - require stronger evidence (3 consecutive)
        required_consecutive = 3
        if consecutive_count >= required_consecutive:
            print(f"REGIME TRANSITION: {_current_stable_regime} → {raw_regime} (required {required_consecutive} consecutive, got {consecutive_count})")
            _current_stable_regime = raw_regime
        else:
            print(f"EXPOSURE PRESERVATION: Maintaining {_current_stable_regime} (need {required_consecutive} consecutive {raw_regime}, have {consecutive_count})")
            return _current_stable_regime
    
    # BEAR → BULL: Require confirmation but not as strict
    elif current_is_bear and new_is_bull:
        required_consecutive = 2  # Standard requirement for bear to bull
        if consecutive_count >= required_consecutive:
            print(f"REGIME TRANSITION: {_current_stable_regime} → {raw_regime} (bear to bull confirmation)")
            _current_stable_regime = raw_regime
        else:
            return _current_stable_regime
    
    # BULL → STRONGER BULL: Allow immediate upgrade
    elif current_is_bull and new_is_bull and raw_regime == "STRONG_BULL" and _current_stable_regime == "BULL":
        print(f"REGIME UPGRADE: {_current_stable_regime} → {raw_regime} (immediate bull upgrade)")
        _current_stable_regime = raw_regime
    
    # Standard transitions: Require minimum consecutive
    elif consecutive_count >= consecutive_needed:
        if raw_regime != _current_stable_regime:
            print(f"REGIME TRANSITION: {_current_stable_regime} → {raw_regime}")
        _current_stable_regime = raw_regime
    
    return _current_stable_regime
    """
    Detect current market regime (BULL, BEAR, SIDEWAYS) for adaptive strategy
    REFINED: Choppiness override for early bull transitions, softened return thresholds, exposure preservation
    
    Args:
        ticker: Stock symbol (if price_data not provided)
        price_data: Historical price data (if ticker not provided)
        lookback_days: Days to analyze for regime detection
        
    Returns:
        Market regime: "BULL", "STRONG_BULL", "BEAR", "STRONG_BEAR", "SIDEWAYS", "CHOPPY", or "CHOPPY_SIDEWAYS"
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
        
        if len(prices) < 2:
            return "SIDEWAYS"  # Need at least 2 data points
        
        # Calculate trend metrics
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        
        if start_price <= 0:
            return "SIDEWAYS"  # Invalid start price
            
        total_return = (end_price - start_price) / start_price
        
        # Calculate moving averages for trend confirmation
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else sma_20
        
        # Calculate volatility (standard deviation of returns)
        returns = prices.pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
        
        # Calculate momentum indicators
        if len(prices) >= 20:
            recent_10_mean = prices.iloc[-10:].mean()
            prev_10_mean = prices.iloc[-20:-10].mean()
            if prev_10_mean > 0:
                recent_momentum = (recent_10_mean - prev_10_mean) / prev_10_mean
            else:
                recent_momentum = 0
        else:
            recent_momentum = 0
        
        # MEDIUM-TERM MOMENTUM for choppiness override (20-30 day window)
        medium_term_return = 0
        if len(prices) >= 25:
            medium_start = prices.iloc[-25]
            medium_end = prices.iloc[-1]
            if medium_start > 0:
                medium_term_return = (medium_end - medium_start) / medium_start
        
        # TREND PERSISTENCE: Check if price is above medium-term moving average
        trend_persistence = end_price > sma_20
        
        # CHOPPY MARKET DETECTION - Critical for avoiding whipsaw losses
        choppy_analysis = detect_choppy_market(prices, lookback_days)
        choppiness_score = choppy_analysis['choppiness_score']
        is_choppy = choppy_analysis['is_choppy']
        is_very_choppy = choppy_analysis['is_very_choppy']
        
        # CHOPPINESS OVERRIDE (Critical): Allow bull behavior when trend and momentum align
        # Prevents delayed re-entry during early bull transitions
        choppy_override = False
        if (is_choppy or is_very_choppy):
            # Check if medium-term momentum is positive AND trend persistence exists
            if medium_term_return > 0.05 and trend_persistence and recent_momentum > 0.02:
                choppy_override = True
                print(f"CHOPPINESS OVERRIDE: Medium-term momentum (+{medium_term_return:.1%}) and trend persistence override choppy classification")
        
        # Apply choppiness classification only if override conditions not met
        if is_very_choppy and not choppy_override:
            return "CHOPPY_SIDEWAYS"
        elif is_choppy and not choppy_override:
            return "CHOPPY"
        
        # REDUCED VOLATILITY OVERRIDE: Only apply in extreme cases and not in clear bull trends
        extreme_volatility_threshold = 1.2  # 120% annualized volatility
        
        if volatility > extreme_volatility_threshold:
            # Check if this is a strong uptrend that deserves volatility tolerance
            if len(prices) >= 20:
                recent_20_start = prices.iloc[-20]
                recent_20_end = prices.iloc[-1]
                if recent_20_start > 0:
                    recent_20_return = (recent_20_end - recent_20_start) / recent_20_start
                else:
                    recent_20_return = 0
            else:
                recent_20_return = total_return
            
            # BULL MARKET VOLATILITY TOLERANCE: Allow much higher volatility in bull trends
            if recent_20_return > 0.15 and end_price > sma_20 and recent_momentum > 0.03:
                # Strong recent bull market with momentum - allow very high volatility
                bull_volatility_threshold = 2.0  # Up to 200% volatility allowed in bull markets
                if volatility <= bull_volatility_threshold:
                    # Continue with normal bull market detection below
                    pass
                else:
                    return "SIDEWAYS"  # Only override if volatility is truly extreme (>200%)
            else:
                # Not a clear bull trend - apply volatility override at lower threshold
                if volatility > 1.5:  # 150% threshold for non-bull trends
                    return "SIDEWAYS"
        
        # SOFTENED RETURN THRESHOLDS: Complement absolute returns with trend persistence and momentum
        # Prevents suppression of slow bull markets during low-volatility grind-ups
        
        # Strong bull: High returns OR strong trend persistence with momentum
        if (total_return > 0.20 and end_price > sma_20 > sma_50 and recent_momentum > 0.05) or \
           (total_return > 0.12 and trend_persistence and recent_momentum > 0.04 and medium_term_return > 0.08):
            return "STRONG_BULL"
        
        # Regular bull: Moderate returns OR trend persistence with positive momentum
        elif (total_return > 0.08 and end_price > sma_20 and recent_momentum > 0.025) or \
             (total_return > 0.04 and trend_persistence and recent_momentum > 0.02 and medium_term_return > 0.04):
            return "BULL"
        
        # Mild bull: Favor participation - low returns but clear trend and momentum
        elif total_return > 0.02 and trend_persistence and recent_momentum > 0.015:
            return "BULL"  # Gradual bullish classification during grind-ups
        
        # Bear markets: Require stronger evidence for bear classification (asymmetric transitions)
        elif total_return < -0.25 and end_price < sma_20 < sma_50 and recent_momentum < -0.05:
            return "STRONG_BEAR"  # Strong downtrend - higher threshold
        elif total_return < -0.15 and end_price < sma_20 and recent_momentum < -0.03:
            return "BEAR"  # Downtrend - higher threshold
        else:
            # EXPOSURE PRESERVATION: When ambiguous, favor maintaining exposure
            # Momentum override for borderline cases - favor bull classification
            if recent_momentum > 0.02 or (trend_persistence and medium_term_return > 0.02):
                return "BULL"  # Default to maintaining exposure when ambiguous
            return "SIDEWAYS"  # Neutral market
            
    except Exception as e:
        print(f"Market regime detection failed: {e}")
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
        print(f"Price volatility calculation failed: {e}")
        return 0.3  # Safe default


def get_adaptive_risk_params(market_regime: str, price_volatility: float = None, strategy: str = "momentum", bull_market_duration: int = 0) -> Dict:
    """
    Get adaptive risk management parameters based on market regime, price volatility, and strategy
    REFACTORED: Volatility no longer reduces conviction in bull regimes - favors market participation over prediction
    
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
    
    # VOLATILITY ADJUSTMENT FACTORS - BULL REGIME PROTECTION
    # Volatility penalties ONLY apply in BEAR regimes - bull regimes maintain full conviction
    if market_regime in ["BULL", "STRONG_BULL"]:
        # Bull regimes: Volatility affects position sizing only, NOT conviction or direction
        vol_stop_multiplier = 1.0    # No volatility penalty on stops in bull markets
        vol_profit_multiplier = 1.0  # No volatility penalty on profit targets in bull markets
        vol_position_multiplier = max(0.7, 1.0 - (price_volatility - 0.3) * 0.3)  # Mild position sizing adjustment only
        vol_trail_multiplier = 1.0   # No volatility penalty on trailing stops in bull markets
        # No volatility warnings in bull regimes - volatility is expected and acceptable
    elif market_regime in ["BEAR", "STRONG_BEAR"]:
        # Bear regimes: Full volatility penalties to protect capital
        if price_volatility > 1.5:  # Extreme volatility (>150%)
            vol_stop_multiplier = 3.0    # Much wider stops
            vol_profit_multiplier = 0.6  # Lower profit targets
            vol_position_multiplier = 0.4  # Much smaller positions
            vol_trail_multiplier = 3.5   # Much wider trailing stops
        elif price_volatility > 1.0:  # High volatility (>100%)
            vol_stop_multiplier = 2.0    # Wider stops
            vol_profit_multiplier = 0.7  # Slightly lower profit targets
            vol_position_multiplier = 0.6  # Smaller positions
            vol_trail_multiplier = 2.5   # Wider trailing stops
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
    else:
        # Neutral regimes: Moderate volatility sensitivity
        vol_stop_multiplier = 1.0 + min((price_volatility - 0.6) * 0.5, 0.5)  # Mild adjustment
        vol_profit_multiplier = max(0.8, 1.0 - (price_volatility - 0.6) * 0.3)
        vol_position_multiplier = max(0.6, 1.0 - (price_volatility - 0.6) * 0.4)
        vol_trail_multiplier = 1.0 + min((price_volatility - 0.6) * 0.8, 0.8)
    
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
            print(f"VALUE STRATEGY OPTIMIZATIONS: Wider stops (+30%), longer holds (3x), larger positions (+20%)")
            get_adaptive_risk_params._value_msg_shown = True
    else:
        # Momentum strategy uses standard multipliers
        value_stop_multiplier = 1.0
        value_profit_multiplier = 1.0
        value_position_multiplier = 1.0
        value_min_hold_multiplier = 1.0
        value_threshold_multiplier = 1.0
    
    # Base parameters by regime with EXPOSURE BIAS - favor market participation in bull regimes
    if market_regime == "STRONG_BULL":
        # Strong bull: Aggressive exposure, minimal exits, assume positive drift
        base_params = {
            'stop_loss_pct': -0.25,      # Wider stops to avoid noise exits
            'take_profit_pct': 3.0,      # High targets to stay invested longer
            'trailing_stop_pct': -0.60,  # Very wide trailing stops
            'position_size_multiplier': 1.5,  # Larger positions for bull participation
            'threshold_tightness': 0.6,  # Tighter thresholds = more trades = more exposure
            'momentum_bias': 0.4,        # Strong bias toward staying invested
            'trend_following': True,     # Enable trend-following mode
            'conviction_multiplier': 2.0, # High conviction in bull markets
            'min_hold_days': 21,         # Longer holds to reduce churn
            'volatility_protection': False, # DISABLED - volatility is acceptable in bull markets
            'bull_duration_days': bull_market_duration,
            'duration_scaling': 1.0,     # No duration scaling - consistent behavior
            'overtrading_protection': False,  # DISABLED - favor exposure over protection
            'exposure_bias': True        # Favor BUY over HOLD when signals are neutral
        }
            
    elif market_regime == "BULL":
        # Regular bull: Strong exposure bias, reduced exits
        base_params = {
            'stop_loss_pct': -0.20,      # Wide stops to avoid noise exits
            'take_profit_pct': 2.0,      # High targets to stay invested
            'trailing_stop_pct': -0.45,  # Wide trailing stops
            'position_size_multiplier': 1.3,  # Larger positions
            'threshold_tightness': 0.7,  # Tighter thresholds for more exposure
            'momentum_bias': 0.3,        # Strong bias toward staying invested
            'trend_following': True,     # Enable trend-following mode
            'conviction_multiplier': 1.7, # High conviction
            'min_hold_days': 14,         # Longer holds to reduce churn
            'volatility_protection': False, # DISABLED - volatility acceptable in bull markets
            'bull_duration_days': bull_market_duration,
            'duration_scaling': 1.0,     # No duration scaling - consistent behavior
            'overtrading_protection': False,  # DISABLED - favor exposure
            'exposure_bias': True        # Favor BUY over HOLD when signals are neutral
        }
            
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
            'min_hold_days': 3,          # Minimum holding period even in bear markets
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
            'min_hold_days': 3,          # Minimum holding period even in bear markets
            'volatility_protection': False, # No volatility override needed
            'bull_duration_days': 0,     # Reset bull market duration
            'duration_scaling': 1.0
        }
    elif market_regime == "CHOPPY":
        # Choppy market: Very conservative parameters to avoid whipsaws
        base_params = {
            'stop_loss_pct': -0.05,      # Tight stop-loss to limit whipsaw damage
            'take_profit_pct': 0.15,     # Lower profit targets
            'trailing_stop_pct': -0.08,  # Tight trailing stop
            'position_size_multiplier': 0.5,  # Much smaller positions
            'threshold_tightness': 2.0,  # Much wider thresholds
            'momentum_bias': 0.0,        # No bias
            'trend_following': False,    # Disable trend-following
            'conviction_multiplier': 0.5, # Lower conviction
            'min_hold_days': 7,          # Longer minimum holding to avoid noise
            'volatility_protection': True, # Enable volatility protection
            'bull_duration_days': 0,     # Reset bull market duration
            'duration_scaling': 1.0,
            'choppy_protection': True    # Enable choppy market protection
        }
    elif market_regime == "CHOPPY_SIDEWAYS":
        # Very choppy market: Extremely conservative, prefer cash
        base_params = {
            'stop_loss_pct': -0.03,      # Very tight stop-loss
            'take_profit_pct': 0.10,     # Very low profit targets
            'trailing_stop_pct': -0.05,  # Very tight trailing stop
            'position_size_multiplier': 0.3,  # Minimal positions
            'threshold_tightness': 3.0,  # Extremely wide thresholds
            'momentum_bias': 0.0,        # No bias
            'trend_following': False,    # Disable trend-following
            'conviction_multiplier': 0.3, # Very low conviction
            'min_hold_days': 10,         # Much longer minimum holding
            'volatility_protection': True, # Enable volatility protection
            'bull_duration_days': 0,     # Reset bull market duration
            'duration_scaling': 1.0,
            'choppy_protection': True,   # Enable choppy market protection
            'avoid_new_positions': True  # Strongly prefer cash in very choppy markets
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
            'min_hold_days': 5,          # Longer minimum holding period to reduce noise
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
    Calculate adaptive buy/sell thresholds based on market regime and score volatility
    REFACTORED: Thresholds clamped within ±10% of base values for consistency, no short-term momentum changes
    
    Args:
        score_history: List of recent Final_Buy_Scores
        market_regime: Current market regime
        lookback: Number of recent scores to analyze
        strategy: Trading strategy ("momentum" or "value") for strategy-specific optimizations
        bull_market_duration: Days in current bull market (IGNORED - no duration scaling for consistency)
        
    Returns:
        Tuple of (buy_threshold, sell_threshold)
    """
    # BASE THRESHOLDS - stable and regime-specific, no complex adaptivity
    if strategy.lower() == "value":
        # VALUE STRATEGY: More conservative, wider spreads
        if market_regime in ["STRONG_BULL", "BULL"]:
            base_buy, base_sell = 5.10, 4.70  # Exposure bias in bull regimes
        elif market_regime in ["STRONG_BEAR", "BEAR"]:
            base_buy, base_sell = 5.50, 4.30  # Conservative in bear regimes
        elif market_regime in ["CHOPPY", "CHOPPY_SIDEWAYS"]:
            base_buy, base_sell = 5.80, 4.00  # Very wide for choppy markets
        else:  # SIDEWAYS
            base_buy, base_sell = 5.25, 4.60  # Moderate
    else:
        # MOMENTUM STRATEGY: Exposure-biased thresholds
        if market_regime in ["STRONG_BULL", "BULL"]:
            base_buy, base_sell = 5.05, 4.75  # Tight spread, exposure bias in bull regimes
        elif market_regime in ["STRONG_BEAR", "BEAR"]:
            base_buy, base_sell = 5.40, 4.40  # Conservative in bear regimes
        elif market_regime in ["CHOPPY", "CHOPPY_SIDEWAYS"]:
            base_buy, base_sell = 5.70, 4.10  # Wide for choppy markets
        else:  # SIDEWAYS
            base_buy, base_sell = 5.20, 4.65  # Moderate
    
    # MINIMAL ADAPTIVITY - only if sufficient history exists
    if len(score_history) >= lookback:
        recent_scores = score_history[-lookback:]
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        # CLAMPED ADJUSTMENT - maximum ±10% deviation from base thresholds
        max_adjustment = 0.1  # 10% maximum adjustment
        
        # Simple volatility adjustment - no complex regime overrides
        volatility_factor = min(std_score * 0.5, max_adjustment)  # Capped at 10%
        
        # Apply minimal adjustment with exposure bias in bull regimes
        if market_regime in ["BULL", "STRONG_BULL"]:
            # Bull regimes: Bias toward lower buy thresholds (more exposure)
            buy_adjustment = -volatility_factor * 0.5  # Favor buying
            sell_adjustment = -volatility_factor * 0.3  # Resist selling
        else:
            # Non-bull regimes: Standard volatility adjustment
            buy_adjustment = volatility_factor
            sell_adjustment = -volatility_factor
        
        # Apply clamped adjustments
        buy_threshold = base_buy + buy_adjustment
        sell_threshold = base_sell + sell_adjustment
        
        # HARD CLAMPS - ensure thresholds stay within ±10% of base values
        buy_threshold = max(base_buy * 0.9, min(buy_threshold, base_buy * 1.1))
        sell_threshold = max(base_sell * 0.9, min(sell_threshold, base_sell * 1.1))
    else:
        # Insufficient history - use base thresholds
        buy_threshold = base_buy
        sell_threshold = base_sell
    
    return buy_threshold, sell_threshold


def calculate_conviction_score(final_buy_score: float, score_history: List[float], market_regime: str) -> float:
    """
    Calculate conviction score based on signal strength and consistency
    REFACTORED: No volatility penalties in bull regimes - volatility is acceptable for market participation
    
    Args:
        final_buy_score: Current Final_Buy_Score
        score_history: Historical scores for trend analysis
        market_regime: Current market regime
        
    Returns:
        Conviction score (0.5 to 2.0) - multiplier for position sizing
    """
    base_conviction = 1.0
    
    # Distance from neutral (5.0) indicates signal strength
    signal_strength = abs(final_buy_score - 5.0)
    conviction_boost = min(signal_strength * 0.3, 0.6)  # Max 0.6 boost
    
    # Trend consistency bonus - NO volatility penalties in bull regimes
    if len(score_history) >= 5:
        recent_scores = score_history[-5:]
        score_volatility = np.std(recent_scores)
        
        # Consistency bonus based on trend direction, not volatility
        if len(recent_scores) >= 3:
            trend_direction = recent_scores[-1] - recent_scores[0]
            if abs(trend_direction) > 0.1:  # Clear trend
                conviction_boost += min(abs(trend_direction) * 0.2, 0.3)  # Max 0.3 boost
        
        # BULL REGIME PROTECTION: No volatility penalties in bull markets
        if market_regime in ["BULL", "STRONG_BULL"]:
            # Volatility is expected and acceptable in bull markets - no penalties
            pass
        elif market_regime in ["BEAR", "STRONG_BEAR"]:
            # Apply volatility penalties only in bear markets for capital protection
            if score_volatility > 0.4:  # High volatility threshold for bear markets
                volatility_penalty = min(score_volatility * 0.3, 0.3)
                conviction_boost -= volatility_penalty
        else:
            # Neutral regimes: Mild volatility consideration
            if score_volatility > 0.5:  # Very high volatility threshold
                volatility_penalty = min(score_volatility * 0.2, 0.2)
                conviction_boost -= volatility_penalty
    
    # Market regime bonus - favor exposure in bull regimes
    if market_regime in ["STRONG_BULL", "BULL"]:
        regime_bonus = 0.3  # Strong conviction boost in bull markets
        conviction_boost += regime_bonus
    elif market_regime in ["STRONG_BEAR"]:
        conviction_boost -= 0.2  # Reduced conviction in strong bear markets only
    
    final_conviction = base_conviction + conviction_boost
    return max(0.5, min(final_conviction, 2.0))  # Clamp between 0.5x and 2.0x


def detect_momentum_reversal(price_data: pd.DataFrame, score_history: List[float], lookback_days: int = 10) -> Dict:
    """
    Detect momentum reversals using both price and sentiment signals
    DISABLED: Reversal logic removed to prevent short-term noise from overriding regime-level decisions
    
    Args:
        price_data: Historical price data
        score_history: Recent Final_Buy_Scores
        lookback_days: Days to analyze for reversal detection
        
    Returns:
        Dictionary with no reversal signals (disabled for consistency)
    """
    # DISABLED: Short-term reversal detection causes more harm than good
    # Favor slow, regime-level confirmation over short-term momentum changes
    return {'reversal_detected': False, 'type': None, 'strength': 0.0, 'reason': 'Reversal detection disabled for consistency'}


def get_trading_recommendation(ticker: str, final_buy_score: float, sentiment_df: pd.DataFrame = None, price_data: pd.DataFrame = None, strategy: str = "momentum") -> Dict:
    """
    Convert Final_Buy_Score into concrete BUY/HOLD/SELL recommendation
    REFINED: Stable regime detection, exposure preservation invariant, drawdown acceptance
    
    Args:
        ticker: Stock symbol
        final_buy_score: Current Final_Buy_Score
        sentiment_df: Historical sentiment data for score history (optional)
        price_data: Historical price data for volatility calculation (optional)
        strategy: Trading strategy ("momentum" or "value") for strategy-specific optimizations
        
    Returns:
        Dictionary with trading recommendation and details
    """
    # Calculate price volatility for risk adjustment (position sizing only in bull regimes)
    price_volatility = calculate_price_volatility(ticker, price_data)
    
    # Use stable regime detection with transition smoothing
    market_regime = get_stable_market_regime(ticker, price_data)
    
    # SENTIMENT-BASED CHOPPINESS OVERRIDE: If regime is choppy but sentiment is strongly positive, allow bull behavior
    # This prevents delayed re-entry during early bull transitions when sentiment leads price action
    if market_regime in ["CHOPPY", "CHOPPY_SIDEWAYS"] and len(score_history) > 0:
        recent_sentiment = np.mean(score_history[-5:]) if len(score_history) >= 5 else final_buy_score
        positive_sentiment_days = len([s for s in score_history[-10:] if s > 5.0]) if len(score_history) >= 10 else (1 if final_buy_score > 5.0 else 0)
        
        # AGGRESSIVE CONDITIONS: Multiple ways to override choppy classification
        if (recent_sentiment > 5.05 and final_buy_score > 5.02) or \
           (positive_sentiment_days >= 6 and final_buy_score > 5.0) or \
           (recent_sentiment > 5.0 and final_buy_score > 5.08):
            print(f"SENTIMENT OVERRIDE: Positive sentiment ({recent_sentiment:.2f}, {positive_sentiment_days}/10 positive days) overrides {market_regime} → BULL behavior")
            market_regime = "BULL"  # Override choppy classification with bull behavior
    
    # Get historical Final_Buy_Scores if available
    score_history = []
    if sentiment_df is not None and hasattr(sentiment_df, 'attrs') and 'Final_Buy_Scores_Over_Time' in sentiment_df.attrs:
        scores_over_time = sentiment_df.attrs['Final_Buy_Scores_Over_Time']
        score_history = [score for _, score in scores_over_time]
    else:
        # Fallback: use current score repeated (not ideal but functional)
        score_history = [final_buy_score] * 10
    
    # Calculate stable adaptive thresholds (clamped within ±10%)
    buy_threshold, sell_threshold = calculate_adaptive_thresholds(score_history, market_regime, strategy=strategy)
    
    # Get risk parameters (volatility affects position sizing only in bull regimes)
    risk_params = get_adaptive_risk_params(market_regime, price_volatility, strategy=strategy)
    
    # Calculate conviction score (no volatility penalties in bull regimes)
    conviction_score = calculate_conviction_score(final_buy_score, score_history, market_regime)
    
    # EXPOSURE PRESERVATION INVARIANT: Check if we're in a bull regime with positive sentiment
    exposure_bias = risk_params.get('exposure_bias', False)
    is_bull_regime = market_regime in ["BULL", "STRONG_BULL"]
    
    # Check sentiment alignment for exposure preservation
    sentiment_positive = len(score_history) > 0 and np.mean(score_history[-5:]) > 5.0
    
    # DRAWDOWN ACCEPTANCE: In bull regimes with positive sentiment, accept drawdowns for participation
    if is_bull_regime and sentiment_positive:
        # Modify thresholds to maintain exposure - prevent premature exits
        original_sell_threshold = sell_threshold
        sell_threshold = min(sell_threshold, 4.3)  # Raise floor to prevent easy exits
        if sell_threshold != original_sell_threshold:
            print(f"EXPOSURE PRESERVATION: Adjusted sell threshold from {original_sell_threshold:.2f} to {sell_threshold:.2f} in {market_regime} with positive sentiment")
    
    # Enhanced decision logic with exposure preservation
    if final_buy_score >= buy_threshold:
        recommendation = "BUY"
        confidence = min(100, ((final_buy_score - buy_threshold) / (10 - buy_threshold)) * 100)
        reasoning = f"Score {final_buy_score:.2f} exceeds {market_regime} buy threshold {buy_threshold:.3f}"
    elif final_buy_score <= sell_threshold:
        # EXPOSURE PRESERVATION: In bull regimes, require stronger bearish evidence
        if is_bull_regime and final_buy_score > 4.2:
            recommendation = "HOLD"  # Resist selling in bull regimes unless very bearish
            confidence = 30
            reasoning = f"Score {final_buy_score:.2f} below sell threshold but exposure preservation prevents sell in {market_regime}"
        else:
            recommendation = "SELL"
            confidence = min(100, ((sell_threshold - final_buy_score) / sell_threshold) * 100)
            reasoning = f"Score {final_buy_score:.2f} below {market_regime} sell threshold {sell_threshold:.3f}"
    else:
        # Neutral zone - apply exposure bias in bull regimes
        if exposure_bias and final_buy_score >= 4.8:  # Lowered from 4.9 for earlier entry
            recommendation = "BUY"  # Favor buying over holding in bull regimes when near neutral
            confidence = 40
            reasoning = f"Score {final_buy_score:.2f} in neutral zone but exposure bias favors buy in {market_regime}"
        # ADDITIONAL EXPOSURE PRESERVATION: If sentiment is trending positive, favor holding over selling
        elif len(score_history) >= 3 and np.mean(score_history[-3:]) > np.mean(score_history[-6:-3]) and final_buy_score > 4.7:
            recommendation = "HOLD"  # Favor holding when sentiment is improving
            confidence = 35
            reasoning = f"Score {final_buy_score:.2f} in neutral zone with improving sentiment trend"
        else:
            recommendation = "HOLD"
            confidence = 100 - abs(final_buy_score - 5.0) * 20  # Closer to neutral = higher confidence
            reasoning = f"Score {final_buy_score:.2f} within {market_regime} hold range ({sell_threshold:.3f} - {buy_threshold:.3f})"
    
    # Apply conviction to confidence
    confidence = min(100, confidence * conviction_score)
    
    return {
        'recommendation': recommendation,
        'confidence': max(0, min(100, confidence)),
        'market_regime': market_regime,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'reasoning': reasoning,
        'risk_params': risk_params,
        'final_buy_score': final_buy_score,
        'conviction_score': conviction_score,
        'position_size_multiplier': risk_params['position_size_multiplier'] * conviction_score,
        'exposure_bias_applied': exposure_bias and recommendation in ["BUY", "HOLD"],
        'exposure_preserved': is_bull_regime and sentiment_positive and recommendation != "SELL"
    }