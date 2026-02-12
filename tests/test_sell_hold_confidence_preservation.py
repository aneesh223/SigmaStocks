"""
Property-based test for SELL/HOLD confidence preservation.

**Property 12: SELL/HOLD Confidence Preservation**
**Validates: Requirements 5.5**

For any SELL or HOLD recommendation, the confidence score MUST remain unchanged 
regardless of the anomaly score value, ensuring microstructure analysis only 
affects BUY decisions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings, assume

from src.logic import get_trading_recommendation


def create_mock_sentiment_df(final_buy_score):
    """Helper to create mock sentiment DataFrame with score history."""
    df = pd.DataFrame({'sentiment': [0.5] * 10})
    df.attrs['Final_Buy_Scores_Over_Time'] = [(i, final_buy_score) for i in range(10)]
    return df


def create_mock_price_data():
    """Helper to create mock price data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1D')
    prices = np.linspace(100, 95, 100)  # Downward trend for SELL
    return pd.DataFrame({
        'Open': prices,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


@given(
    anomaly_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    final_buy_score=st.floats(min_value=0.0, max_value=4.5, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_sell_confidence_preservation(anomaly_score, final_buy_score):
    """
    **Property 12: SELL/HOLD Confidence Preservation (SELL)**
    **Validates: Requirements 5.5**
    
    For any SELL recommendation, the confidence should remain unchanged 
    regardless of anomaly score.
    """
    mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
    mock_price_data = create_mock_price_data()
    
    mock_microstructure = {
        'anomaly_score': anomaly_score,
        'status': 'Test status',
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
    
    # Only test if recommendation is SELL
    if baseline_result['recommendation'] == 'SELL':
        baseline_confidence = baseline_result['confidence']
        adjusted_confidence = adjusted_result['confidence']
        
        # CRITICAL: Confidence should be unchanged for SELL
        assert abs(adjusted_confidence - baseline_confidence) < 0.01, \
            f"SELL confidence should remain unchanged at {baseline_confidence:.2f} " \
            f"regardless of anomaly score {anomaly_score:.3f}, got {adjusted_confidence:.2f}"
        
        # Verify recommendation is still SELL
        assert adjusted_result['recommendation'] == 'SELL', \
            f"Recommendation should remain SELL, got {adjusted_result['recommendation']}"


@given(
    anomaly_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    final_buy_score=st.floats(min_value=4.6, max_value=5.1, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_hold_confidence_preservation(anomaly_score, final_buy_score):
    """
    **Property 12: SELL/HOLD Confidence Preservation (HOLD)**
    **Validates: Requirements 5.5**
    
    For any HOLD recommendation, the confidence should remain unchanged 
    regardless of anomaly score.
    """
    mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
    mock_price_data = create_mock_price_data()
    
    mock_microstructure = {
        'anomaly_score': anomaly_score,
        'status': 'Test status',
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
    
    # Only test if recommendation is HOLD
    if baseline_result['recommendation'] == 'HOLD':
        baseline_confidence = baseline_result['confidence']
        adjusted_confidence = adjusted_result['confidence']
        
        # CRITICAL: Confidence should be unchanged for HOLD
        assert abs(adjusted_confidence - baseline_confidence) < 0.01, \
            f"HOLD confidence should remain unchanged at {baseline_confidence:.2f} " \
            f"regardless of anomaly score {anomaly_score:.3f}, got {adjusted_confidence:.2f}"
        
        # Verify recommendation is still HOLD
        assert adjusted_result['recommendation'] == 'HOLD', \
            f"Recommendation should remain HOLD, got {adjusted_result['recommendation']}"


def test_sell_confidence_preservation_extreme_anomalies():
    """Test SELL confidence preservation with extreme anomaly scores."""
    mock_sentiment_df = create_mock_sentiment_df(3.0)  # Low score for SELL
    mock_price_data = create_mock_price_data()
    
    # Test with extreme high anomaly (critical void)
    mock_microstructure_high = {
        'anomaly_score': 0.95,
        'status': 'Critical liquidity void detected',
        'confidence': 'high'
    }
    
    # Test with extreme low anomaly (accumulation)
    mock_microstructure_low = {
        'anomaly_score': 0.05,
        'status': 'Strong accumulation zone detected',
        'confidence': 'high'
    }
    
    # Get baseline
    with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
        baseline_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=3.0,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    if baseline_result['recommendation'] == 'SELL':
        baseline_confidence = baseline_result['confidence']
        
        # Test with high anomaly
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_high):
            result_high = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=3.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Test with low anomaly
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_low):
            result_low = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=3.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Both should preserve confidence
        assert abs(result_high['confidence'] - baseline_confidence) < 0.01, \
            f"SELL confidence should be preserved with high anomaly"
        
        assert abs(result_low['confidence'] - baseline_confidence) < 0.01, \
            f"SELL confidence should be preserved with low anomaly"
        
        # Both should remain SELL
        assert result_high['recommendation'] == 'SELL'
        assert result_low['recommendation'] == 'SELL'


def test_hold_confidence_preservation_extreme_anomalies():
    """Test HOLD confidence preservation with extreme anomaly scores."""
    mock_sentiment_df = create_mock_sentiment_df(4.9)  # Neutral score for HOLD
    mock_price_data = create_mock_price_data()
    
    # Test with extreme high anomaly (critical void)
    mock_microstructure_high = {
        'anomaly_score': 0.95,
        'status': 'Critical liquidity void detected',
        'confidence': 'high'
    }
    
    # Test with extreme low anomaly (accumulation)
    mock_microstructure_low = {
        'anomaly_score': 0.05,
        'status': 'Strong accumulation zone detected',
        'confidence': 'high'
    }
    
    # Get baseline
    with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
        baseline_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=4.9,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    if baseline_result['recommendation'] == 'HOLD':
        baseline_confidence = baseline_result['confidence']
        
        # Test with high anomaly
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_high):
            result_high = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=4.9,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Test with low anomaly
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_low):
            result_low = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=4.9,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Both should preserve confidence
        assert abs(result_high['confidence'] - baseline_confidence) < 0.01, \
            f"HOLD confidence should be preserved with high anomaly"
        
        assert abs(result_low['confidence'] - baseline_confidence) < 0.01, \
            f"HOLD confidence should be preserved with low anomaly"
        
        # Both should remain HOLD
        assert result_high['recommendation'] == 'HOLD'
        assert result_low['recommendation'] == 'HOLD'
