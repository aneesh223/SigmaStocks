"""
Property-based test for BUY confidence boost with accumulation zone detection.

**Property 11: BUY Confidence Boost for Accumulation**
**Validates: Requirements 5.4**

For any BUY recommendation with anomaly score < 0.2, the adjusted confidence 
MUST be approximately 120% of the original confidence (confidence_adjusted = 
confidence_original * 1.2), within floating-point rounding tolerance.
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
    anomaly_score=st.floats(min_value=0.0, max_value=0.199, allow_nan=False, allow_infinity=False),
    final_buy_score=st.floats(min_value=5.2, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_buy_confidence_boost_accumulation(anomaly_score, final_buy_score):
    """
    **Property 11: BUY Confidence Boost for Accumulation**
    **Validates: Requirements 5.4**
    
    For any BUY recommendation with anomaly score < 0.2, the confidence 
    should be boosted to approximately 120% of the original.
    """
    mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
    mock_price_data = create_mock_price_data()
    
    mock_microstructure = {
        'anomaly_score': anomaly_score,
        'status': 'Strong accumulation zone detected',
        'confidence': 'high'
    }
    
    # Get baseline without microstructure adjustment
    with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
        baseline_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=final_buy_score,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # Get recommendation with microstructure adjustment
    with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
        adjusted_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=final_buy_score,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # Only test if baseline recommendation is BUY
    if baseline_result['recommendation'] == 'BUY':
        baseline_confidence = baseline_result['confidence']
        adjusted_confidence = adjusted_result['confidence']
        
        # Calculate expected confidence (120% of original)
        expected_confidence = baseline_confidence * 1.2
        
        # Clamp to 100 if it exceeds
        expected_confidence = min(expected_confidence, 100)
        
        # Allow for small floating-point tolerance (±0.5%)
        tolerance = baseline_confidence * 0.005
        
        assert abs(adjusted_confidence - expected_confidence) <= tolerance, \
            f"BUY with anomaly score {anomaly_score:.3f} should boost confidence from " \
            f"{baseline_confidence:.2f} to ~{expected_confidence:.2f}, got {adjusted_confidence:.2f}"
        
        # Verify confidence is still in valid range [0, 100]
        assert 0 <= adjusted_confidence <= 100, \
            f"Adjusted confidence {adjusted_confidence} is outside valid range [0, 100]"
        
        # Verify microstructure status is in reasoning
        assert 'Microstructure' in adjusted_result['reasoning'], \
            "Microstructure status should be included in reasoning"


@given(
    anomaly_score=st.floats(min_value=0.0, max_value=0.199, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_buy_confidence_bounds_after_accumulation_boost(anomaly_score):
    """
    **Property 11: BUY Confidence Boost for Accumulation (Bounds Check)**
    **Validates: Requirements 5.4, 5.7**
    
    After applying accumulation boost, confidence must remain in [0, 100].
    Even if boosted by 20%, it should be clamped to 100.
    """
    for base_confidence_score in [5.2, 7.0, 10.0]:
        mock_sentiment_df = create_mock_sentiment_df(base_confidence_score)
        mock_price_data = create_mock_price_data()
        
        mock_microstructure = {
            'anomaly_score': anomaly_score,
            'status': 'Strong accumulation zone detected',
            'confidence': 'high'
        }
        
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
            result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=base_confidence_score,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        if result['recommendation'] == 'BUY':
            confidence = result['confidence']
            
            # CRITICAL: Confidence must be in valid range
            assert 0 <= confidence <= 100, \
                f"Confidence {confidence} is outside valid range [0, 100] after accumulation boost"


def test_buy_confidence_accumulation_boundary():
    """Test exact boundary at anomaly score = 0.2."""
    mock_sentiment_df = create_mock_sentiment_df(6.0)
    mock_price_data = create_mock_price_data()
    
    # Test with score exactly at 0.2 (should NOT trigger accumulation boost)
    mock_microstructure_at_boundary = {
        'anomaly_score': 0.2,
        'status': 'Normal market conditions',
        'confidence': 'high'
    }
    
    # Test with score just below 0.2 (should trigger accumulation boost)
    mock_microstructure_below_boundary = {
        'anomaly_score': 0.199,
        'status': 'Strong accumulation zone detected',
        'confidence': 'high'
    }
    
    # Get baseline
    with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
        baseline_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=6.0,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    if baseline_result['recommendation'] == 'BUY':
        baseline_confidence = baseline_result['confidence']
        
        # Test at boundary (0.2) - should NOT apply 20% boost
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_at_boundary):
            result_at_boundary = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Test below boundary (0.199) - should apply 20% boost
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_below_boundary):
            result_below_boundary = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        confidence_at_boundary = result_at_boundary['confidence']
        confidence_below_boundary = result_below_boundary['confidence']
        
        expected_boosted = min(baseline_confidence * 1.2, 100)
        
        # At 0.2, should NOT apply boost
        assert abs(confidence_at_boundary - baseline_confidence) < 1.0, \
            f"At boundary (0.2), confidence should be ~{baseline_confidence:.2f}, got {confidence_at_boundary:.2f}"
        
        # Below 0.2, should apply 20% boost
        assert abs(confidence_below_boundary - expected_boosted) < 1.0, \
            f"Below boundary (0.199), confidence should be ~{expected_boosted:.2f}, got {confidence_below_boundary:.2f}"


def test_buy_confidence_accumulation_clamping():
    """Test that confidence is clamped to 100 when boost would exceed it."""
    # Use a high buy score that will result in high baseline confidence
    mock_sentiment_df = create_mock_sentiment_df(9.0)
    mock_price_data = create_mock_price_data()
    
    mock_microstructure = {
        'anomaly_score': 0.1,
        'status': 'Strong accumulation zone detected',
        'confidence': 'high'
    }
    
    # Get baseline
    with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
        baseline_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=9.0,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # Get boosted result
    with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
        boosted_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=9.0,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    if baseline_result['recommendation'] == 'BUY':
        baseline_confidence = baseline_result['confidence']
        boosted_confidence = boosted_result['confidence']
        
        # If baseline is high enough that 120% would exceed 100, verify clamping
        if baseline_confidence > 83.33:  # 83.33 * 1.2 = 100
            assert boosted_confidence == 100, \
                f"Confidence should be clamped to 100 when boost exceeds limit, got {boosted_confidence}"
        else:
            expected = baseline_confidence * 1.2
            assert abs(boosted_confidence - expected) < 1.0, \
                f"Confidence should be boosted to {expected:.2f}, got {boosted_confidence}"
