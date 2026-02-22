"""
Property-based test for confidence bounds invariant.

**Property 13: Confidence Bounds Invariant**
**Validates: Requirements 5.7**

For any trading recommendation after microstructure adjustment, the final 
confidence score MUST be clamped to the range [0, 100], preventing invalid 
confidence values from propagating through the system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings, assume

from src.logic import get_trading_recommendation


def create_mock_sentiment_df(final_buy_score):
    """Helper to create mock sentiment DataFrame with score history."""
    df = pd.DataFrame({'sentiment': [0.8] * 10})
    df.attrs['Final_Buy_Scores_Over_Time'] = [(i, final_buy_score) for i in range(10)]
    return df


def create_mock_price_data():
    """Helper to create mock price data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1D')
    prices = np.linspace(100, 120, 100)
    return pd.DataFrame({
        'Open': prices,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


@given(
    anomaly_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    final_buy_score=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    recommendation_type=st.sampled_from(['BUY', 'SELL', 'HOLD'])
)
@settings(max_examples=200, deadline=None)
def test_confidence_bounds_invariant(anomaly_score, final_buy_score, recommendation_type):
    """
    **Property 13: Confidence Bounds Invariant**
    **Validates: Requirements 5.7**
    
    For ANY trading recommendation with ANY anomaly score and ANY final_buy_score,
    the final confidence MUST be in [0, 100].
    """
    mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
    mock_price_data = create_mock_price_data()
    
    # Determine appropriate status based on anomaly score
    if anomaly_score > 0.85:
        status = 'Critical liquidity void detected'
    elif anomaly_score > 0.70:
        status = 'High microstructure volatility'
    elif anomaly_score >= 0.20:
        status = 'Normal market conditions'
    else:
        status = 'Strong accumulation zone detected'
    
    mock_microstructure = {
        'anomaly_score': anomaly_score,
        'status': status,
        'confidence': 'high'
    }
    
    with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
        result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=final_buy_score,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    confidence = result['confidence']
    
    # CRITICAL: Confidence MUST be in valid range [0, 100]
    assert 0 <= confidence <= 100, \
        f"Confidence {confidence} is outside valid range [0, 100] " \
        f"(recommendation={result['recommendation']}, anomaly_score={anomaly_score:.3f}, " \
        f"final_buy_score={final_buy_score:.2f})"
    
    # Verify confidence is a valid number (not NaN or Inf)
    assert not np.isnan(confidence), \
        f"Confidence is NaN (recommendation={result['recommendation']}, " \
        f"anomaly_score={anomaly_score:.3f}, final_buy_score={final_buy_score:.2f})"
    
    assert not np.isinf(confidence), \
        f"Confidence is Inf (recommendation={result['recommendation']}, " \
        f"anomaly_score={anomaly_score:.3f}, final_buy_score={final_buy_score:.2f})"


@given(
    anomaly_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_confidence_bounds_extreme_buy_scores(anomaly_score):
    """
    **Property 13: Confidence Bounds Invariant (Extreme Scores)**
    **Validates: Requirements 5.7**
    
    Test with extreme buy scores that might produce very high or very low confidence.
    """
    # Test with extreme high buy score (should produce high confidence)
    for extreme_score in [0.0, 10.0, 15.0]:
        mock_sentiment_df = create_mock_sentiment_df(extreme_score)
        mock_price_data = create_mock_price_data()
        
        if anomaly_score > 0.85:
            status = 'Critical liquidity void detected'
        elif anomaly_score > 0.70:
            status = 'High microstructure volatility'
        elif anomaly_score >= 0.20:
            status = 'Normal market conditions'
        else:
            status = 'Strong accumulation zone detected'
        
        mock_microstructure = {
            'anomaly_score': anomaly_score,
            'status': status,
            'confidence': 'high'
        }
        
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
            result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=extreme_score,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        confidence = result['confidence']
        
        # CRITICAL: Even with extreme scores, confidence must be bounded
        assert 0 <= confidence <= 100, \
            f"Confidence {confidence} is outside valid range [0, 100] with extreme score {extreme_score}"


def test_confidence_bounds_with_accumulation_boost():
    """Test that accumulation boost (1.2x) doesn't exceed 100."""
    # Use a very high buy score to get high baseline confidence
    mock_sentiment_df = create_mock_sentiment_df(9.5)
    mock_price_data = create_mock_price_data()
    
    # Low anomaly score triggers 20% boost
    mock_microstructure = {
        'anomaly_score': 0.05,
        'status': 'Strong accumulation zone detected',
        'confidence': 'high'
    }
    
    with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
        result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=9.5,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    if result['recommendation'] == 'BUY':
        confidence = result['confidence']
        
        # Even with 20% boost, confidence must not exceed 100
        assert confidence <= 100, \
            f"Confidence {confidence} exceeds 100 after accumulation boost"
        
        assert confidence >= 0, \
            f"Confidence {confidence} is negative"


def test_confidence_bounds_with_multiple_adjustments():
    """Test confidence bounds with various adjustment scenarios."""
    test_cases = [
        # (final_buy_score, anomaly_score, expected_adjustment)
        (9.0, 0.9, 'high_anomaly_reduction'),  # High score + high anomaly
        (9.0, 0.1, 'accumulation_boost'),      # High score + accumulation
        (6.0, 0.75, 'moderate_anomaly'),       # Medium score + moderate anomaly
        (10.0, 0.05, 'extreme_boost'),         # Extreme score + accumulation
        (5.2, 0.95, 'buy_with_critical'),      # Low buy + critical anomaly
    ]
    
    for final_buy_score, anomaly_score, scenario in test_cases:
        mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
        mock_price_data = create_mock_price_data()
        
        if anomaly_score > 0.85:
            status = 'Critical liquidity void detected'
        elif anomaly_score > 0.70:
            status = 'High microstructure volatility'
        elif anomaly_score >= 0.20:
            status = 'Normal market conditions'
        else:
            status = 'Strong accumulation zone detected'
        
        mock_microstructure = {
            'anomaly_score': anomaly_score,
            'status': status,
            'confidence': 'high'
        }
        
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
            result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=final_buy_score,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        confidence = result['confidence']
        
        # CRITICAL: All scenarios must produce bounded confidence
        assert 0 <= confidence <= 100, \
            f"Scenario '{scenario}' produced confidence {confidence} outside [0, 100] " \
            f"(score={final_buy_score}, anomaly={anomaly_score:.2f})"


def test_confidence_bounds_with_zero_baseline():
    """Test that confidence bounds hold even if baseline confidence is very low."""
    # Use a score that might produce low confidence
    mock_sentiment_df = create_mock_sentiment_df(5.2)
    mock_price_data = create_mock_price_data()
    
    # High anomaly reduces confidence by 30%
    mock_microstructure = {
        'anomaly_score': 0.9,
        'status': 'Critical liquidity void detected',
        'confidence': 'high'
    }
    
    with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
        result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=5.2,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    confidence = result['confidence']
    
    # Even with reduction, confidence must not go negative
    assert confidence >= 0, \
        f"Confidence {confidence} is negative after reduction"
    
    assert confidence <= 100, \
        f"Confidence {confidence} exceeds 100"
