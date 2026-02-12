"""
Property-based test for BUY confidence adjustment with moderate anomaly scores.

**Property 10: BUY Confidence Adjustment for Moderate Anomaly**
**Validates: Requirements 5.3**

For any BUY recommendation with anomaly score between 0.7 and 0.8, the adjusted 
confidence MUST be approximately 85% of the original confidence (confidence_adjusted = 
confidence_original * 0.85), within floating-point rounding tolerance.
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
    anomaly_score=st.floats(min_value=0.701, max_value=0.8, allow_nan=False, allow_infinity=False),
    final_buy_score=st.floats(min_value=5.2, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_buy_confidence_reduction_moderate_anomaly(anomaly_score, final_buy_score):
    """
    **Property 10: BUY Confidence Adjustment for Moderate Anomaly**
    **Validates: Requirements 5.3**
    
    For any BUY recommendation with anomaly score between 0.7 and 0.8, 
    the confidence should be reduced to approximately 85% of the original.
    """
    mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
    mock_price_data = create_mock_price_data()
    
    mock_microstructure = {
        'anomaly_score': anomaly_score,
        'status': 'High microstructure volatility',
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
        
        # Calculate expected confidence (85% of original)
        expected_confidence = baseline_confidence * 0.85
        
        # Allow for small floating-point tolerance (±0.5%)
        tolerance = baseline_confidence * 0.005
        
        assert abs(adjusted_confidence - expected_confidence) <= tolerance, \
            f"BUY with anomaly score {anomaly_score:.3f} should reduce confidence from " \
            f"{baseline_confidence:.2f} to ~{expected_confidence:.2f}, got {adjusted_confidence:.2f}"
        
        # Verify confidence is still in valid range [0, 100]
        assert 0 <= adjusted_confidence <= 100, \
            f"Adjusted confidence {adjusted_confidence} is outside valid range [0, 100]"
        
        # Verify microstructure status is in reasoning
        assert 'Microstructure' in adjusted_result['reasoning'], \
            "Microstructure status should be included in reasoning"


@given(
    anomaly_score=st.floats(min_value=0.701, max_value=0.8, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_buy_confidence_bounds_after_moderate_anomaly_adjustment(anomaly_score):
    """
    **Property 10: BUY Confidence Adjustment for Moderate Anomaly (Bounds Check)**
    **Validates: Requirements 5.3, 5.7**
    
    After applying moderate anomaly adjustment, confidence must remain in [0, 100].
    """
    for base_confidence_score in [5.2, 7.0, 10.0]:
        mock_sentiment_df = create_mock_sentiment_df(base_confidence_score)
        mock_price_data = create_mock_price_data()
        
        mock_microstructure = {
            'anomaly_score': anomaly_score,
            'status': 'High microstructure volatility',
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
                f"Confidence {confidence} is outside valid range [0, 100] after moderate anomaly adjustment"


def test_buy_confidence_moderate_anomaly_boundaries():
    """Test exact boundaries at anomaly scores 0.7 and 0.8."""
    mock_sentiment_df = create_mock_sentiment_df(6.0)
    mock_price_data = create_mock_price_data()
    
    # Test with score exactly at 0.7 (should NOT trigger moderate anomaly adjustment)
    mock_microstructure_at_lower_boundary = {
        'anomaly_score': 0.7,
        'status': 'Normal market conditions',
        'confidence': 'high'
    }
    
    # Test with score just above 0.7 (should trigger moderate anomaly adjustment)
    mock_microstructure_above_lower_boundary = {
        'anomaly_score': 0.701,
        'status': 'High microstructure volatility',
        'confidence': 'high'
    }
    
    # Test with score exactly at 0.8 (should trigger moderate anomaly adjustment)
    mock_microstructure_at_upper_boundary = {
        'anomaly_score': 0.8,
        'status': 'High microstructure volatility',
        'confidence': 'high'
    }
    
    # Test with score just above 0.8 (should trigger high anomaly adjustment instead)
    mock_microstructure_above_upper_boundary = {
        'anomaly_score': 0.801,
        'status': 'Critical liquidity void detected',
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
        
        # Test at lower boundary (0.7) - should NOT apply 15% reduction
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_at_lower_boundary):
            result_at_lower = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Test above lower boundary (0.701) - should apply 15% reduction
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_above_lower_boundary):
            result_above_lower = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Test at upper boundary (0.8) - should apply 15% reduction
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_at_upper_boundary):
            result_at_upper = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Test above upper boundary (0.801) - should apply 30% reduction
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_above_upper_boundary):
            result_above_upper = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        confidence_at_lower = result_at_lower['confidence']
        confidence_above_lower = result_above_lower['confidence']
        confidence_at_upper = result_at_upper['confidence']
        confidence_above_upper = result_above_upper['confidence']
        
        expected_moderate = baseline_confidence * 0.85  # 15% reduction
        expected_high = baseline_confidence * 0.7  # 30% reduction
        
        # At 0.7, should NOT apply any reduction
        assert abs(confidence_at_lower - baseline_confidence) < 1.0, \
            f"At boundary (0.7), confidence should be ~{baseline_confidence:.2f}, got {confidence_at_lower:.2f}"
        
        # Above 0.7, should apply 15% reduction
        assert abs(confidence_above_lower - expected_moderate) < 1.0, \
            f"Above lower boundary (0.701), confidence should be ~{expected_moderate:.2f}, got {confidence_above_lower:.2f}"
        
        # At 0.8, should apply 15% reduction
        assert abs(confidence_at_upper - expected_moderate) < 1.0, \
            f"At upper boundary (0.8), confidence should be ~{expected_moderate:.2f}, got {confidence_at_upper:.2f}"
        
        # Above 0.8, should apply 30% reduction
        assert abs(confidence_above_upper - expected_high) < 1.0, \
            f"Above upper boundary (0.801), confidence should be ~{expected_high:.2f}, got {confidence_above_upper:.2f}"
