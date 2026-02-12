"""
Property-based test for fail-safe execution.

**Property 2: Fail-Safe Execution**
**Validates: Requirements 5.6, 6.2, 6.3, 6.4, 6.5**

For any trading recommendation inputs, the get_trading_recommendation function 
MUST return a valid recommendation dictionary even if analyze_liquidity raises 
an exception or returns None. The system shall default to the original confidence 
score in all failure modes.
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
    final_buy_score=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    exception_type=st.sampled_from([
        Exception("Generic error"),
        RuntimeError("Runtime error"),
        ValueError("Value error"),
        KeyError("Key error"),
        AttributeError("Attribute error"),
        TypeError("Type error"),
        ImportError("Import error"),
    ])
)
@settings(max_examples=100, deadline=None)
def test_failsafe_with_exceptions(final_buy_score, exception_type):
    """
    **Property 2: Fail-Safe Execution (Exception Handling)**
    **Validates: Requirements 5.6, 6.2, 6.5**
    
    When analyze_liquidity raises ANY exception, get_trading_recommendation 
    MUST still return a valid dictionary with unchanged confidence.
    """
    mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
    mock_price_data = create_mock_price_data()
    
    # Get baseline without microstructure
    with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
        baseline_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=final_buy_score,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # Test with exception
    with patch('src.microstructure.analyze_liquidity', side_effect=exception_type):
        result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=final_buy_score,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # CRITICAL: Function must return a valid dictionary
    assert isinstance(result, dict), \
        f"get_trading_recommendation must return dict even with exception, got {type(result)}"
    
    # CRITICAL: Must have required keys
    required_keys = ['recommendation', 'confidence', 'reasoning']
    for key in required_keys:
        assert key in result, \
            f"Result missing required key '{key}' after exception {type(exception_type).__name__}"
    
    # CRITICAL: Recommendation must be valid
    assert result['recommendation'] in ['BUY', 'SELL', 'HOLD'], \
        f"Invalid recommendation '{result['recommendation']}' after exception"
    
    # CRITICAL: Confidence must be in valid range
    assert 0 <= result['confidence'] <= 100, \
        f"Confidence {result['confidence']} outside [0, 100] after exception"
    
    # CRITICAL: Confidence should be unchanged (within small tolerance for floating point)
    assert abs(result['confidence'] - baseline_result['confidence']) < 0.01, \
        f"Confidence changed from {baseline_result['confidence']:.2f} to {result['confidence']:.2f} " \
        f"after exception {type(exception_type).__name__}"


@given(
    final_buy_score=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_failsafe_with_none_return(final_buy_score):
    """
    **Property 2: Fail-Safe Execution (None Return)**
    **Validates: Requirements 5.6, 6.4**
    
    When analyze_liquidity returns None, get_trading_recommendation 
    MUST still return a valid dictionary with unchanged confidence.
    """
    mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
    mock_price_data = create_mock_price_data()
    
    # Get baseline
    with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
        baseline_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=final_buy_score,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # Test with None return
    with patch('src.microstructure.analyze_liquidity', return_value=None):
        result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=final_buy_score,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # CRITICAL: Function must return a valid dictionary
    assert isinstance(result, dict), \
        "get_trading_recommendation must return dict even when analyze_liquidity returns None"
    
    # CRITICAL: Confidence should be unchanged
    assert abs(result['confidence'] - baseline_result['confidence']) < 0.01, \
        f"Confidence changed from {baseline_result['confidence']:.2f} to {result['confidence']:.2f} " \
        f"when analyze_liquidity returns None"


@given(
    final_buy_score=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_failsafe_with_malformed_return(final_buy_score):
    """
    **Property 2: Fail-Safe Execution (Malformed Return)**
    **Validates: Requirements 5.6, 6.5**
    
    When analyze_liquidity returns malformed data, get_trading_recommendation 
    MUST still return a valid dictionary with unchanged confidence.
    """
    mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
    mock_price_data = create_mock_price_data()
    
    # Get baseline
    with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
        baseline_result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=final_buy_score,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # Test with various malformed returns
    malformed_returns = [
        {},  # Empty dict
        {'anomaly_score': 'invalid'},  # Wrong type
        {'status': 'test'},  # Missing anomaly_score
        {'anomaly_score': None, 'status': 'test'},  # None score
        {'anomaly_score': float('nan'), 'status': 'test'},  # NaN score
        {'anomaly_score': float('inf'), 'status': 'test'},  # Inf score
    ]
    
    for malformed_data in malformed_returns:
        with patch('src.microstructure.analyze_liquidity', return_value=malformed_data):
            result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=final_buy_score,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # CRITICAL: Function must return a valid dictionary
        assert isinstance(result, dict), \
            f"get_trading_recommendation must return dict with malformed data {malformed_data}"
        
        # CRITICAL: Must have required keys
        assert 'recommendation' in result and 'confidence' in result, \
            f"Result missing required keys with malformed data {malformed_data}"
        
        # CRITICAL: Confidence must be valid
        assert 0 <= result['confidence'] <= 100, \
            f"Confidence {result['confidence']} outside [0, 100] with malformed data {malformed_data}"


def test_failsafe_with_import_error():
    """Test that system works even if microstructure module can't be imported."""
    mock_sentiment_df = create_mock_sentiment_df(6.0)
    mock_price_data = create_mock_price_data()
    
    # Simulate import error
    with patch('src.microstructure.analyze_liquidity', side_effect=ImportError("Module not found")):
        result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=6.0,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # CRITICAL: System must continue to work
    assert isinstance(result, dict)
    assert result['recommendation'] in ['BUY', 'SELL', 'HOLD']
    assert 0 <= result['confidence'] <= 100


def test_failsafe_preserves_recommendation():
    """Test that exceptions don't change the recommendation type."""
    test_cases = [
        (9.0, 'BUY'),   # High score should be BUY
        (3.0, 'SELL'),  # Low score should be SELL
        (4.9, 'HOLD'),  # Neutral score should be HOLD
    ]
    
    for final_buy_score, expected_rec in test_cases:
        mock_sentiment_df = create_mock_sentiment_df(final_buy_score)
        mock_price_data = create_mock_price_data()
        
        # Get baseline
        with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
            baseline_result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=final_buy_score,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Test with exception
        with patch('src.microstructure.analyze_liquidity', side_effect=RuntimeError("Test error")):
            result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=final_buy_score,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # CRITICAL: Recommendation should not change due to exception
        assert result['recommendation'] == baseline_result['recommendation'], \
            f"Recommendation changed from {baseline_result['recommendation']} to {result['recommendation']} " \
            f"due to exception (score={final_buy_score})"


def test_failsafe_with_network_timeout():
    """Test fail-safe behavior with network-like timeouts."""
    mock_sentiment_df = create_mock_sentiment_df(6.0)
    mock_price_data = create_mock_price_data()
    
    # Simulate timeout
    with patch('src.microstructure.analyze_liquidity', side_effect=TimeoutError("Network timeout")):
        result = get_trading_recommendation(
            ticker='TEST',
            final_buy_score=6.0,
            sentiment_df=mock_sentiment_df,
            price_data=mock_price_data,
            strategy='momentum'
        )
    
    # CRITICAL: System must handle timeout gracefully
    assert isinstance(result, dict)
    assert result['recommendation'] in ['BUY', 'SELL', 'HOLD']
    assert 0 <= result['confidence'] <= 100


def test_failsafe_multiple_consecutive_failures():
    """Test that system remains stable across multiple consecutive failures."""
    mock_sentiment_df = create_mock_sentiment_df(6.0)
    mock_price_data = create_mock_price_data()
    
    # Simulate multiple consecutive calls with failures
    for i in range(5):
        with patch('src.microstructure.analyze_liquidity', side_effect=Exception(f"Error {i}")):
            result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # CRITICAL: Each call must succeed despite failures
        assert isinstance(result, dict), f"Call {i} failed to return dict"
        assert result['recommendation'] in ['BUY', 'SELL', 'HOLD'], f"Call {i} returned invalid recommendation"
        assert 0 <= result['confidence'] <= 100, f"Call {i} returned invalid confidence"
