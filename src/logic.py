"""
Trading logic module for regime detection and risk management.
Contains market regime classification, adaptive thresholds, and position sizing logic.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import yfinance as yf

# Regime classification constants
BULLISH_REGIMES = {"BULL", "STRONG_BULL"}
BEARISH_REGIMES = {"BEAR", "STRONG_BEAR"}
CHOPPY_REGIMES = {"CHOPPY", "CHOPPY_SIDEWAYS"}
NEUTRAL_REGIMES = {"SIDEWAYS"}

logger = logging.getLogger(__name__)


def calculate_bull_market_duration(price_data: pd.DataFrame, current_regime: str, lookback_days: int = 120) -> int:
    """Calculate duration of current bull market in days."""
    if current_regime not in BULLISH_REGIMES:
        return 0
    
    if len(price_data) < lookback_days:
        return 0
    
    recent_data = price_data.tail(lookback_days)
    prices = recent_data['Close']
    
    # Calculate 20-day rolling returns
    rolling_returns = prices.pct_change(20).dropna()
    
    # Find bull market start (first day with sustained positive momentum)
    bull_start_idx = None
    for i in range(len(rolling_returns) - 1, -1, -1):
        if rolling_returns.iloc[i] < 0.05:
            bull_start_idx = i + 1
            break
    
    if bull_start_idx is None:
        return lookback_days
    else:
        return len(rolling_returns) - bull_start_idx


def calculate_adaptive_profit_target(entry_price: float, current_price: float, bull_market_duration: int, 
                                   base_profit_pct: float, market_regime: str) -> float:
    """Calculate profit target scaled by bull market momentum and duration."""
    if market_regime not in BULLISH_REGIMES or entry_price <= 0:
        return base_profit_pct
    
    current_gain_pct = (current_price - entry_price) / entry_price
    
    # Scale target based on bull market duration
    duration_multiplier = min(1.0 + (bull_market_duration / 60.0), 2.5)
    
    # Scale based on current unrealized gains
    if current_gain_pct > 0.1:
        momentum_multiplier = min(1.0 + (current_gain_pct * 2), 2.0)
    else:
        momentum_multiplier = 1.0
    
    # Regime-specific scaling
    regime_multiplier = 1.5 if market_regime == "STRONG_BULL" else 1.2
    
    adaptive_target = base_profit_pct * duration_multiplier * momentum_multiplier * regime_multiplier
    max_target = 3.0 if market_regime == "STRONG_BULL" else 2.0
    
    return min(adaptive_target, max_target)


def detect_choppy_market(prices: pd.Series, lookback_days: int = 30) -> dict:
    """Detect choppy/sideways market conditions using vectorized calculations."""
    if len(prices) < max(10, lookback_days // 2):
        return {'is_choppy': False, 'choppiness_score': 0.0, 'reason': 'Insufficient data'}
    
    recent_prices = prices.tail(lookback_days)
    returns = recent_prices.pct_change().dropna()
    
    if len(returns) < 5:
        return {'is_choppy': False, 'choppiness_score': 0.0, 'reason': 'Insufficient returns data'}
    
    # Directional movement calculation
    up_moves = returns[returns > 0].sum()
    down_moves = abs(returns[returns < 0].sum())
    total_movement = up_moves + down_moves
    
    directional_index = abs(up_moves - down_moves) / total_movement if total_movement > 0 else 0
    
    # Price oscillation analysis using vectorized operations
    sma_10 = recent_prices.rolling(10).mean()
    price_above_sma = recent_prices > sma_10
    crossovers = (price_above_sma != price_above_sma.shift(1)).sum()
    crossover_rate = crossovers / len(recent_prices)
    
    # Volatility vs trend analysis
    volatility = returns.std() * (252 ** 0.5)
    net_return = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
    efficiency_ratio = abs(net_return) / volatility if volatility > 0 else 0
    
    # Consecutive reversals using vectorized operations
    direction_changes = (returns.shift(1) * returns < 0).sum()
    reversal_rate = direction_changes / len(returns)
    
    # Range-bound behavior
    price_range = (recent_prices.max() - recent_prices.min()) / recent_prices.mean()
    
    # Choppiness scoring
    choppiness_factors = {
        'low_directional_movement': max(0, 0.3 - directional_index) * 3.33,
        'high_crossover_rate': min(1, crossover_rate * 5),
        'low_efficiency': max(0, 0.1 - efficiency_ratio) * 10,
        'high_reversal_rate': min(1, reversal_rate * 2),
        'narrow_range': max(0, 0.15 - price_range) * 6.67 if price_range < 0.15 else 0
    }
    
    weights = {
        'low_directional_movement': 0.25,
        'high_crossover_rate': 0.25,
        'low_efficiency': 0.20,
        'high_reversal_rate': 0.20,
        'narrow_range': 0.10
    }
    
    choppiness_score = sum(choppiness_factors[factor] * weights[factor] 
                          for factor in choppiness_factors)
    
    is_choppy = choppiness_score > 0.6
    is_very_choppy = choppiness_score > 0.8
    
    main_factors = [factor for factor, score in choppiness_factors.items() if score > 0.3]
    
    if is_very_choppy:
        reason = f"Very choppy: {', '.join(main_factors[:2])}"
    elif is_choppy:
        reason = f"Choppy: {', '.join(main_factors[:2])}"
    else:
        reason = "Trending: Directional movement detected"
    
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


def detect_market_regime(ticker: str = None, price_data: pd.DataFrame = None, lookback_days: int = 30) -> str:
    """
    Detect current market regime using price trends and momentum indicators.
    
    Args:
        ticker: Stock symbol (if price_data not provided)
        price_data: Historical price data (if ticker not provided)
        lookback_days: Days to analyze for regime detection
        
    Returns:
        Market regime classification
    """
    # Get price data if not provided
    if price_data is None:
        if ticker is None:
            return "SIDEWAYS"
        stock = yf.Ticker(ticker)
        price_data = stock.history(period=f"{lookback_days + 10}d")
    
    if len(price_data) < lookback_days:
        return "SIDEWAYS"
    
    recent_data = price_data.tail(lookback_days)
    prices = recent_data['Close']
    
    if len(prices) < 2:
        return "SIDEWAYS"
    
    # Calculate trend metrics
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    
    if start_price <= 0:
        return "SIDEWAYS"
        
    total_return = (end_price - start_price) / start_price
    
    # Calculate moving averages for trend confirmation
    sma_20 = prices.rolling(20).mean().iloc[-1]
    sma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else sma_20
    
    # Calculate volatility
    returns = prices.pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5)
    
    # Calculate momentum indicators
    if len(prices) >= 20:
        recent_10_mean = prices.iloc[-10:].mean()
        prev_10_mean = prices.iloc[-20:-10].mean()
        recent_momentum = (recent_10_mean - prev_10_mean) / prev_10_mean if prev_10_mean > 0 else 0
    else:
        recent_momentum = 0
    
    # Medium-term momentum for choppiness override
    medium_term_return = 0
    if len(prices) >= 25:
        medium_start = prices.iloc[-25]
        medium_end = prices.iloc[-1]
        if medium_start > 0:
            medium_term_return = (medium_end - medium_start) / medium_start
    
    trend_persistence = end_price > sma_20
    
    # Choppy market detection
    choppy_analysis = detect_choppy_market(prices, lookback_days)
    choppiness_score = choppy_analysis['choppiness_score']
    is_choppy = choppy_analysis['is_choppy']
    is_very_choppy = choppy_analysis['is_very_choppy']
    
    # Choppiness override for early bull transitions
    choppy_override = False
    if (is_choppy or is_very_choppy):
        if medium_term_return > 0.05 and trend_persistence and recent_momentum > 0.02:
            choppy_override = True
            logger.debug(f"Choppiness override: medium-term momentum +{medium_term_return:.1%}")
    
    # Apply choppiness classification
    if is_very_choppy and not choppy_override:
        return "CHOPPY_SIDEWAYS"
    elif is_choppy and not choppy_override:
        return "CHOPPY"
    
    # Volatility override for extreme cases
    extreme_volatility_threshold = 1.2
    
    if volatility > extreme_volatility_threshold:
        if len(prices) >= 20:
            recent_20_start = prices.iloc[-20]
            recent_20_end = prices.iloc[-1]
            recent_20_return = (recent_20_end - recent_20_start) / recent_20_start if recent_20_start > 0 else 0
        else:
            recent_20_return = total_return
        
        # Allow higher volatility in bull trends
        if recent_20_return > 0.15 and end_price > sma_20 and recent_momentum > 0.03:
            bull_volatility_threshold = 2.0
            if volatility > bull_volatility_threshold:
                return "SIDEWAYS"
        else:
            if volatility > 1.5:
                return "SIDEWAYS"
    
    # Regime classification based on returns and momentum
    if (total_return > 0.20 and end_price > sma_20 > sma_50 and recent_momentum > 0.05) or \
       (total_return > 0.12 and trend_persistence and recent_momentum > 0.04 and medium_term_return > 0.08):
        return "STRONG_BULL"
    
    elif (total_return > 0.08 and end_price > sma_20 and recent_momentum > 0.025) or \
         (total_return > 0.04 and trend_persistence and recent_momentum > 0.02 and medium_term_return > 0.04):
        return "BULL"
    
    elif total_return > 0.02 and trend_persistence and recent_momentum > 0.015:
        return "BULL"
    
    elif total_return < -0.25 and end_price < sma_20 < sma_50 and recent_momentum < -0.05:
        return "STRONG_BEAR"
    elif total_return < -0.15 and end_price < sma_20 and recent_momentum < -0.03:
        return "BEAR"
    else:
        # Default to maintaining exposure when ambiguous
        if recent_momentum > 0.02 or (trend_persistence and medium_term_return > 0.02):
            return "BULL"
        return "SIDEWAYS"


# Per-asset regime state tracking
_asset_regime_states = {}

class AssetRegimeState:
    """Per-asset regime state to prevent cross-contamination."""
    def __init__(self):
        self.regime_history = []
        self.current_stable_regime = None
    
    def update_regime(self, raw_regime: str) -> str:
        """Update regime with transition smoothing."""
        if self.current_stable_regime is None:
            self.current_stable_regime = raw_regime
            self.regime_history = [raw_regime]
            return raw_regime
        
        # Maintain history of last 5 evaluations
        self.regime_history.append(raw_regime)
        if len(self.regime_history) > 5:
            self.regime_history.pop(0)
        
        consecutive_needed = 2
        consecutive_count = 0
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i] == raw_regime:
                consecutive_count += 1
            else:
                break
        
        # Asymmetric transition rules
        current_is_bull = self.current_stable_regime in BULLISH_REGIMES
        new_is_bull = raw_regime in BULLISH_REGIMES
        current_is_bear = self.current_stable_regime in BEARISH_REGIMES
        new_is_bear = raw_regime in BEARISH_REGIMES
        
        # Require stronger evidence to exit bull regimes
        if current_is_bull and not new_is_bull:
            required_consecutive = 3
            if consecutive_count >= required_consecutive:
                logger.debug(f"Regime transition: {self.current_stable_regime} → {raw_regime}")
                self.current_stable_regime = raw_regime
            else:
                logger.debug(f"Maintaining {self.current_stable_regime} (need {required_consecutive} consecutive)")
                return self.current_stable_regime
        
        # Standard bear to bull transition
        elif current_is_bear and new_is_bull:
            required_consecutive = 2
            if consecutive_count >= required_consecutive:
                logger.debug(f"Regime transition: {self.current_stable_regime} → {raw_regime}")
                self.current_stable_regime = raw_regime
            else:
                return self.current_stable_regime
        
        # Allow immediate bull upgrades
        elif current_is_bull and new_is_bull and raw_regime == "STRONG_BULL" and self.current_stable_regime == "BULL":
            logger.debug(f"Regime upgrade: {self.current_stable_regime} → {raw_regime}")
            self.current_stable_regime = raw_regime
        
        # Standard transitions
        elif consecutive_count >= consecutive_needed:
            if raw_regime != self.current_stable_regime:
                logger.debug(f"Regime transition: {self.current_stable_regime} → {raw_regime}")
            self.current_stable_regime = raw_regime
        
        return self.current_stable_regime

def _get_asset_regime_state(ticker: str) -> AssetRegimeState:
    """Get or create per-asset regime state."""
    if ticker not in _asset_regime_states:
        _asset_regime_states[ticker] = AssetRegimeState()
    return _asset_regime_states[ticker]

def get_stable_market_regime(ticker: str = None, price_data: pd.DataFrame = None, lookback_days: int = 30) -> str:
    """Get market regime with transition smoothing to prevent flip-flopping."""
    raw_regime = detect_market_regime(ticker, price_data, lookback_days)
    
    # Use ticker-specific state or fallback identifier
    asset_key = ticker if ticker else "price_data_asset"
    asset_state = _get_asset_regime_state(asset_key)
    
    return asset_state.update_regime(raw_regime)


def calculate_price_volatility(ticker: str = None, price_data: pd.DataFrame = None, lookback_days: int = 30) -> float:
    """Calculate annualized price volatility."""
    # Get price data if not provided
    if price_data is None:
        if ticker is None:
            return 0.3
        stock = yf.Ticker(ticker)
        price_data = stock.history(period=f"{lookback_days + 5}d")
    
    if len(price_data) < lookback_days:
        return 0.3
    
    recent_data = price_data.tail(lookback_days)
    prices = recent_data['Close']
    
    # Calculate annualized volatility
    returns = prices.pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5)
    
    return max(0.1, min(volatility, 3.0))


def get_adaptive_risk_params(market_regime: str, price_volatility: float = None, 
                           strategy: str = "momentum", bull_market_duration: int = 0) -> Dict:
    """Get risk management parameters based on market regime and strategy."""
    if price_volatility is None:
        price_volatility = 0.3
    
    # Volatility adjustments - only penalize in bear regimes
    if market_regime in BULLISH_REGIMES:
        vol_stop_multiplier = 1.0
        vol_profit_multiplier = 1.0
        vol_position_multiplier = max(0.7, 1.0 - (price_volatility - 0.3) * 0.3)
        vol_trail_multiplier = 1.0
    elif market_regime in BEARISH_REGIMES:
        if price_volatility > 1.5:
            vol_stop_multiplier, vol_profit_multiplier = 3.0, 0.6
            vol_position_multiplier, vol_trail_multiplier = 0.4, 3.5
        elif price_volatility > 1.0:
            vol_stop_multiplier, vol_profit_multiplier = 2.0, 0.7
            vol_position_multiplier, vol_trail_multiplier = 0.6, 2.5
        elif price_volatility > 0.6:
            vol_stop_multiplier, vol_profit_multiplier = 1.5, 0.85
            vol_position_multiplier, vol_trail_multiplier = 0.8, 1.8
        else:
            vol_stop_multiplier = vol_profit_multiplier = vol_position_multiplier = vol_trail_multiplier = 1.0
    else:
        vol_stop_multiplier = 1.0 + min((price_volatility - 0.6) * 0.5, 0.5)
        vol_profit_multiplier = max(0.8, 1.0 - (price_volatility - 0.6) * 0.3)
        vol_position_multiplier = max(0.6, 1.0 - (price_volatility - 0.6) * 0.4)
        vol_trail_multiplier = 1.0 + min((price_volatility - 0.6) * 0.8, 0.8)
    
    # Strategy-specific adjustments
    if strategy.lower() == "value":
        value_multipliers = {
            'stop': 1.3, 'profit': 0.7, 'position': 1.2, 
            'min_hold': 3.0, 'threshold': 1.4
        }
        logger.debug("Value strategy optimizations applied")
    else:
        value_multipliers = {
            'stop': 1.0, 'profit': 1.0, 'position': 1.0, 
            'min_hold': 1.0, 'threshold': 1.0
        }
    
    # Base parameters by regime
    regime_params = {
        "STRONG_BULL": {
            'stop_loss_pct': -0.25, 'take_profit_pct': 3.0, 'trailing_stop_pct': -0.60,
            'position_size_multiplier': 1.5, 'threshold_tightness': 0.6,
            'momentum_bias': 0.4, 'conviction_multiplier': 2.0, 'min_hold_days': 21,
            'trend_following': True, 'exposure_bias': True
        },
        "BULL": {
            'stop_loss_pct': -0.20, 'take_profit_pct': 2.0, 'trailing_stop_pct': -0.45,
            'position_size_multiplier': 1.3, 'threshold_tightness': 0.7,
            'momentum_bias': 0.3, 'conviction_multiplier': 1.7, 'min_hold_days': 14,
            'trend_following': True, 'exposure_bias': True
        },
        "STRONG_BEAR": {
            'stop_loss_pct': -0.04, 'take_profit_pct': 0.08, 'trailing_stop_pct': -0.05,
            'position_size_multiplier': 0.4, 'threshold_tightness': 2.2,
            'momentum_bias': -0.3, 'conviction_multiplier': 0.7, 'min_hold_days': 3,
            'trend_following': False, 'exposure_bias': False
        },
        "BEAR": {
            'stop_loss_pct': -0.06, 'take_profit_pct': 0.12, 'trailing_stop_pct': -0.07,
            'position_size_multiplier': 0.6, 'threshold_tightness': 1.5,
            'momentum_bias': -0.15, 'conviction_multiplier': 0.8, 'min_hold_days': 3,
            'trend_following': False, 'exposure_bias': False
        },
        "CHOPPY": {
            'stop_loss_pct': -0.05, 'take_profit_pct': 0.15, 'trailing_stop_pct': -0.08,
            'position_size_multiplier': 0.5, 'threshold_tightness': 2.0,
            'momentum_bias': 0.0, 'conviction_multiplier': 0.5, 'min_hold_days': 7,
            'trend_following': False, 'choppy_protection': True
        },
        "CHOPPY_SIDEWAYS": {
            'stop_loss_pct': -0.03, 'take_profit_pct': 0.10, 'trailing_stop_pct': -0.05,
            'position_size_multiplier': 0.3, 'threshold_tightness': 3.0,
            'momentum_bias': 0.0, 'conviction_multiplier': 0.3, 'min_hold_days': 10,
            'trend_following': False, 'choppy_protection': True, 'avoid_new_positions': True
        }
    }
    
    # Default to sideways if regime not found
    base_params = regime_params.get(market_regime, {
        'stop_loss_pct': -0.08, 'take_profit_pct': 0.25, 'trailing_stop_pct': -0.10,
        'position_size_multiplier': 1.0, 'threshold_tightness': 1.0,
        'momentum_bias': 0.0, 'conviction_multiplier': 1.0, 'min_hold_days': 5,
        'trend_following': False
    })
    
    # Apply adjustments
    adjusted_params = base_params.copy()
    adjusted_params['stop_loss_pct'] = max(base_params['stop_loss_pct'] * vol_stop_multiplier * value_multipliers['stop'], -0.50)
    adjusted_params['take_profit_pct'] = max(base_params['take_profit_pct'] * vol_profit_multiplier * value_multipliers['profit'], 0.05)
    adjusted_params['trailing_stop_pct'] = max(base_params['trailing_stop_pct'] * vol_trail_multiplier * value_multipliers['stop'], -0.60)
    adjusted_params['position_size_multiplier'] = max(base_params['position_size_multiplier'] * vol_position_multiplier * value_multipliers['position'], 0.2)
    adjusted_params['threshold_tightness'] = base_params['threshold_tightness'] * value_multipliers['threshold']
    adjusted_params['min_hold_days'] = int(base_params['min_hold_days'] * value_multipliers['min_hold'])
    
    # Bull regime aggression floor
    if market_regime in BULLISH_REGIMES:
        min_position_multiplier = 1.0 if market_regime == "BULL" else 1.2
        min_conviction_multiplier = 1.5 if market_regime == "BULL" else 1.8
        
        adjusted_params['position_size_multiplier'] = max(adjusted_params['position_size_multiplier'], min_position_multiplier)
        adjusted_params['conviction_multiplier'] = max(adjusted_params['conviction_multiplier'], min_conviction_multiplier)
        adjusted_params['threshold_tightness'] = min(adjusted_params['threshold_tightness'], 1.0)
    
    # Add metadata
    adjusted_params.update({
        'price_volatility': price_volatility,
        'strategy': strategy,
        'bull_duration_days': bull_market_duration,
        'volatility_adjustment': {
            'stop_multiplier': vol_stop_multiplier,
            'profit_multiplier': vol_profit_multiplier,
            'position_multiplier': vol_position_multiplier,
            'trail_multiplier': vol_trail_multiplier
        }
    })
    
    return adjusted_params


def calculate_adaptive_thresholds(score_history: List[float], market_regime: str, lookback: int = 20, 
                                strategy: str = "momentum", bull_market_duration: int = 0) -> Tuple[float, float]:
    """Calculate adaptive buy/sell thresholds based on market regime and score volatility."""
    # Base thresholds by strategy and regime
    if strategy.lower() == "value":
        regime_thresholds = {
            ("STRONG_BULL", "BULL"): (5.10, 4.70),
            ("STRONG_BEAR", "BEAR"): (5.50, 4.30),
            ("CHOPPY", "CHOPPY_SIDEWAYS"): (5.80, 4.00),
            "default": (5.25, 4.60)
        }
    else:
        regime_thresholds = {
            ("STRONG_BULL", "BULL"): (5.05, 4.75),
            ("STRONG_BEAR", "BEAR"): (5.40, 4.40),
            ("CHOPPY", "CHOPPY_SIDEWAYS"): (5.70, 4.10),
            "default": (5.20, 4.65)
        }
    
    # Find base thresholds
    base_buy = base_sell = None
    for regimes, (buy, sell) in regime_thresholds.items():
        if regimes != "default" and market_regime in regimes:
            base_buy, base_sell = buy, sell
            break
    
    if base_buy is None:
        base_buy, base_sell = regime_thresholds["default"]
    
    # Apply minimal adaptivity if sufficient history
    if len(score_history) >= lookback:
        recent_scores = score_history[-lookback:]
        std_score = np.std(recent_scores)
        
        # Clamped adjustment (max ±10%)
        volatility_factor = min(std_score * 0.5, 0.1)
        
        if market_regime in BULLISH_REGIMES:
            buy_adjustment = -volatility_factor * 0.5  # Favor buying
            sell_adjustment = -volatility_factor * 0.3  # Resist selling
        else:
            buy_adjustment = volatility_factor
            sell_adjustment = -volatility_factor
        
        buy_threshold = base_buy + buy_adjustment
        sell_threshold = base_sell + sell_adjustment
        
        # Hard clamps within ±10% of base values
        buy_threshold = max(base_buy * 0.9, min(buy_threshold, base_buy * 1.1))
        sell_threshold = max(base_sell * 0.9, min(sell_threshold, base_sell * 1.1))
    else:
        buy_threshold = base_buy
        sell_threshold = base_sell
    
    return buy_threshold, sell_threshold


def calculate_conviction_score(final_buy_score: float, score_history: List[float], market_regime: str) -> float:
    """Calculate conviction score based on signal strength and consistency."""
    base_conviction = 1.0
    
    # Signal strength from distance to neutral
    signal_strength = abs(final_buy_score - 5.0)
    conviction_boost = min(signal_strength * 0.3, 0.6)
    
    # Trend consistency bonus
    if len(score_history) >= 5:
        recent_scores = score_history[-5:]
        score_volatility = np.std(recent_scores)
        
        # Trend direction bonus
        if len(recent_scores) >= 3:
            trend_direction = recent_scores[-1] - recent_scores[0]
            if abs(trend_direction) > 0.1:
                conviction_boost += min(abs(trend_direction) * 0.2, 0.3)
        
        # Volatility penalties only in bear regimes
        if market_regime in BEARISH_REGIMES:
            if score_volatility > 0.4:
                volatility_penalty = min(score_volatility * 0.3, 0.3)
                conviction_boost -= volatility_penalty
        elif market_regime not in BULLISH_REGIMES:
            # Neutral regimes: mild volatility consideration
            if score_volatility > 0.5:
                volatility_penalty = min(score_volatility * 0.2, 0.2)
                conviction_boost -= volatility_penalty
    
    # Regime-based adjustments
    if market_regime in BULLISH_REGIMES:
        conviction_boost += 0.3  # Bull market bonus
    elif market_regime == "STRONG_BEAR":
        conviction_boost -= 0.2  # Bear market penalty
    
    final_conviction = base_conviction + conviction_boost
    return max(0.5, min(final_conviction, 2.0))


def detect_momentum_reversal(price_data: pd.DataFrame, score_history: List[float], lookback_days: int = 10) -> Dict:
    """Detect momentum reversals - disabled for consistency."""
    # Reversal logic disabled to prevent short-term noise from overriding regime decisions
    return {
        'reversal_detected': False, 
        'type': None, 
        'strength': 0.0, 
        'reason': 'Reversal detection disabled for consistency'
    }


def get_trading_recommendation(ticker: str, final_buy_score: float, sentiment_df: pd.DataFrame = None, 
                             price_data: pd.DataFrame = None, strategy: str = "momentum") -> Dict:
    """
    Convert Final_Buy_Score into concrete BUY/HOLD/SELL recommendation.
    
    Args:
        ticker: Stock symbol
        final_buy_score: Current Final_Buy_Score
        sentiment_df: Historical sentiment data for score history
        price_data: Historical price data for volatility calculation
        strategy: Trading strategy ("momentum" or "value")
        
    Returns:
        Dictionary with trading recommendation and details
    """
    # Calculate price volatility for risk adjustment
    price_volatility = calculate_price_volatility(ticker, price_data)
    
    # Use stable regime detection with transition smoothing
    market_regime = get_stable_market_regime(ticker, price_data)
    
    # Get historical scores if available
    if sentiment_df is not None and hasattr(sentiment_df, 'attrs') and 'Final_Buy_Scores_Over_Time' in sentiment_df.attrs:
        scores_over_time = sentiment_df.attrs['Final_Buy_Scores_Over_Time']
        score_history = [score for _, score in scores_over_time]
    else:
        score_history = [final_buy_score] * 10
    
    # Sentiment-based choppiness override
    if market_regime in CHOPPY_REGIMES and len(score_history) > 0:
        recent_sentiment = np.mean(score_history[-5:]) if len(score_history) >= 5 else final_buy_score
        positive_sentiment_days = len([s for s in score_history[-10:] if s > 5.0]) if len(score_history) >= 10 else (1 if final_buy_score > 5.0 else 0)
        
        # Override choppy classification with bull behavior if sentiment is strong
        if (recent_sentiment > 5.05 and final_buy_score > 5.02) or \
           (positive_sentiment_days >= 6 and final_buy_score > 5.0) or \
           (recent_sentiment > 5.0 and final_buy_score > 5.08):
            logger.debug(f"Sentiment override: {recent_sentiment:.2f} sentiment overrides {market_regime}")
            market_regime = "BULL"
    
    # Calculate adaptive thresholds and risk parameters
    buy_threshold, sell_threshold = calculate_adaptive_thresholds(score_history, market_regime, strategy=strategy)
    risk_params = get_adaptive_risk_params(market_regime, price_volatility, strategy=strategy)
    conviction_score = calculate_conviction_score(final_buy_score, score_history, market_regime)
    
    # Exposure preservation logic
    exposure_bias = risk_params.get('exposure_bias', False)
    is_bull_regime = market_regime in BULLISH_REGIMES
    sentiment_positive = len(score_history) > 0 and np.mean(score_history[-5:]) > 5.0
    
    # Adjust sell threshold in bull regimes to maintain exposure
    if is_bull_regime and sentiment_positive:
        original_sell_threshold = sell_threshold
        sell_threshold = min(sell_threshold, 4.3)
        if sell_threshold != original_sell_threshold:
            logger.debug(f"Exposure preservation: adjusted sell threshold from {original_sell_threshold:.2f} to {sell_threshold:.2f}")
    
    # Generate recommendation
    if final_buy_score >= buy_threshold:
        recommendation = "BUY"
        confidence = min(100, ((final_buy_score - buy_threshold) / (10 - buy_threshold)) * 100)
        reasoning = f"Score {final_buy_score:.2f} exceeds {market_regime} buy threshold {buy_threshold:.3f}"
    elif final_buy_score <= sell_threshold:
        # Resist selling in bull regimes unless very bearish
        if is_bull_regime and final_buy_score > 4.2:
            recommendation = "HOLD"
            confidence = 30
            reasoning = f"Score {final_buy_score:.2f} below sell threshold but exposure preservation prevents sell"
        else:
            recommendation = "SELL"
            confidence = min(100, ((sell_threshold - final_buy_score) / sell_threshold) * 100)
            reasoning = f"Score {final_buy_score:.2f} below {market_regime} sell threshold {sell_threshold:.3f}"
    else:
        # Neutral zone logic
        if exposure_bias and final_buy_score >= 4.8:
            recommendation = "BUY"
            confidence = 40
            reasoning = f"Score {final_buy_score:.2f} in neutral zone but exposure bias favors buy"
        elif len(score_history) >= 3 and np.mean(score_history[-3:]) > np.mean(score_history[-6:-3]) and final_buy_score > 4.7:
            recommendation = "HOLD"
            confidence = 35
            reasoning = f"Score {final_buy_score:.2f} in neutral zone with improving sentiment trend"
        else:
            recommendation = "HOLD"
            confidence = 100 - abs(final_buy_score - 5.0) * 20
            reasoning = f"Score {final_buy_score:.2f} within {market_regime} hold range"
    
    # Apply conviction to confidence
    confidence = min(100, confidence * conviction_score)
    
    # Apply microstructure adjustment (real-time only - requires intraday data)
    microstructure_applied = False
    try:
        from src.microstructure import analyze_liquidity
        
        microstructure = analyze_liquidity(ticker)
        anomaly_score = microstructure['anomaly_score']
        
        # Only apply if we have high-confidence data (not fallback neutral score)
        if microstructure['confidence'] == 'high':
            microstructure_applied = True
            
            # Adjust BUY recommendations based on microstructure
            if recommendation == "BUY":
                original_confidence = confidence
                if anomaly_score > 0.85:
                    # Critical liquidity void - reduce confidence by 30%
                    confidence *= 0.7
                    reasoning += f" | Microstructure: {microstructure['status']} (score: {anomaly_score:.2f})"
                    logger.info(f"Microstructure adjustment applied for {ticker}: confidence {original_confidence:.1f} -> {confidence:.1f} (anomaly score: {anomaly_score:.2f})")
                elif anomaly_score > 0.70:
                    # High volatility - reduce confidence by 15%
                    confidence *= 0.85
                    reasoning += f" | Microstructure: {microstructure['status']} (score: {anomaly_score:.2f})"
                    logger.info(f"Microstructure adjustment applied for {ticker}: confidence {original_confidence:.1f} -> {confidence:.1f} (anomaly score: {anomaly_score:.2f})")
                elif anomaly_score < 0.20:
                    # Strong accumulation - boost confidence by 20%
                    confidence *= 1.2
                    reasoning += f" | Microstructure: {microstructure['status']} (score: {anomaly_score:.2f})"
                    logger.info(f"Microstructure adjustment applied for {ticker}: confidence {original_confidence:.1f} -> {confidence:.1f} (anomaly score: {anomaly_score:.2f})")
                
                # Ensure confidence stays in [0, 100]
                confidence = max(0, min(100, confidence))
            
            # Also adjust SELL recommendations for extreme anomalies
            elif recommendation == "SELL" and anomaly_score > 0.85:
                # Critical liquidity void confirms sell signal - boost confidence
                original_confidence = confidence
                confidence *= 1.15
                reasoning += f" | Microstructure: {microstructure['status']} confirms sell (score: {anomaly_score:.2f})"
                logger.info(f"Microstructure confirmation for {ticker}: confidence {original_confidence:.1f} -> {confidence:.1f}")
                confidence = max(0, min(100, confidence))
    
    except Exception as e:
        # Fail gracefully - don't break existing logic
        # Note: Microstructure analysis requires intraday data and won't work for historical backtesting
        logger.debug(f"Microstructure analysis skipped: {e}")
    
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
        'exposure_preserved': is_bull_regime and sentiment_positive and recommendation != "SELL",
        'microstructure_applied': microstructure_applied
    }