"""
Property-based test for BUY confidence adjustment with high anomaly scores.

**Property 9: BUY Confidence Adjustment for High Anomaly**
**Validates: Requirements 5.2**

For any BUY recommendation with anomaly score > 0.8, the adjusted confidence 
MUST be approximately 70% of the original confidence (confidence_adjusted = 
confidence_original * 0.7), within floating-point rounding tolerance.
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
    # Add score history that would lead to BUY recommendation
    df.attrs['Final_Buy_Scores_Over_Time'] = [(i, final_buy_score) for i in range(10)]
    return df


def create_mock_price_data():
    """Helper to create mock price data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1D')
    prices = np.linspace(100, 120, 100)  # Upward trend
    return pd.DataFrame({
        'Open': prices,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


@given(
    anomaly_score=st.floats(min_value=0.801, max_value=1.0, allow_nan=False, allow_infinity=False),
    final_buy_score=st.floats(min_value=5.2, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_buy_confidence_reduction_high_anomaly(anomaly_score, final_buy_score):
    """
    **Property 9: BUY Confidence Adjustment for High Anomaly**
    **Validates: Requirements 5.2**
    
    For any BUY recommendation with anomaly score > 0.8, the confidence 
    should be reduced to approximately 70% of the original.
    """
    # Create mock data
    mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
    mock_price_data = create_mock_price_data()
    
    # Mock analyze_liquidity to return high anomaly score
    mock_microstructure = {
        'anomaly_score': anomaly_score,
        'status': 'Critical liquidity void detected',
        'confidence': 'high'
    }
    
    # Get recommendation without microstructure adjustment (to get baseline)
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
        
        # Calculate expected confidence (70% of original)
        expected_confidence = baseline_confidence * 0.7
        
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
    anomaly_score=st.floats(min_value=0.801, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_buy_confidence_bounds_after_high_anomaly_adjustment(anomaly_score):
    """
    **Property 9: BUY Confidence Adjustment for High Anomaly (Bounds Check)**
    **Validates: Requirements 5.2, 5.7**
    
    After applying high anomaly adjustment, confidence must remain in [0, 100].
    """
    # Test with extreme confidence values
    for base_confidence_score in [5.2, 7.0, 10.0]:
        mock_sentiment_df = create_mock_sentiment_df(base_confidence_score)
        mock_price_data = create_mock_price_data()
        
        mock_microstructure = {
            'anomaly_score': anomaly_score,
            'status': 'Critical liquidity void detected',
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
                f"Confidence {confidence} is outside valid range [0, 100] after high anomaly adjustment"


def test_buy_confidence_high_anomaly_boundary():
    """Test exact boundary at anomaly score = 0.8 (should NOT trigger high anomaly adjustment)."""
    mock_sentiment_df = create_mock_sentiment_df(6.0)
    mock_price_data = create_mock_price_data()
    
    # Test with score exactly at 0.8 (should not trigger > 0.8 adjustment)
    mock_microstructure_at_boundary = {
        'anomaly_score': 0.8,
        'status': 'High microstructure volatility',
        'confidence': 'high'
    }
    
    # Test with score just above 0.8 (should trigger > 0.8 adjustment)
    mock_microstructure_above_boundary = {
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
    
    # Test at boundary (0.8)
    with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_at_boundary):
        result_at_boundary = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=6.0,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # Test above boundary (0.801)
    with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_above_boundary):
        result_above_boundary = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=6.0,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    if baseline_result['recommendation'] == 'BUY':
        baseline_confidence = baseline_result['confidence']
        
        # At exactly 0.8, should NOT apply 30% reduction (should apply 15% reduction instead)
        confidence_at_boundary = result_at_boundary['confidence']
        expected_at_boundary = baseline_confidence * 0.85  # 15% reduction for 0.7 < score <= 0.8
        
        # Above 0.8, should apply 30% reduction
        confidence_above_boundary = result_above_boundary['confidence']
        expected_above_boundary = baseline_confidence * 0.7
        
        # Verify the boundary behavior
        assert abs(confidence_at_boundary - expected_at_boundary) < 1.0, \
            f"At boundary (0.8), confidence should be ~{expected_at_boundary:.2f}, got {confidence_at_boundary:.2f}"
        
        assert abs(confidence_above_boundary - expected_above_boundary) < 1.0, \
            f"Above boundary (0.801), confidence should be ~{expected_above_boundary:.2f}, got {confidence_above_boundary:.2f}"
