"""
Property-based test for analyze_liquidity output range invariant.

**Property 1: Output Range Invariant**
**Validates: Requirements 4.4**

For any ticker symbol and market conditions, the anomaly_score returned by 
analyze_liquidity MUST always be a float value within the closed interval [0.0, 1.0],
regardless of input data outliers or extreme market conditions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from datetime import datetime, timedelta

from src.microstructure import analyze_liquidity


# Custom strategy for generating OHLCV DataFrames with various conditions
@st.composite
def ohlcv_dataframe_strategy(draw, min_rows=10, max_rows=500):
    """
    Generate random OHLCV DataFrames with various market conditions.
    
    This includes:
    - Normal market conditions
    - Extreme volatility
    - Zero volume periods
    - Price gaps
    - Outliers
    """
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    # Generate base price with wide range to test extremes
    base_price = draw(st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False))
    
    # Generate volatility (can be very high to test extreme conditions)
    volatility = draw(st.floats(min_value=0.0, max_value=base_price * 2.0, allow_nan=False, allow_infinity=False))
    
    # Generate base volume (including possibility of zero volume)
    base_volume = draw(st.integers(min_value=0, max_value=100000000))
    
    # Create timestamps
    start_date = datetime(2024, 1, 1, 9, 30)
    dates = [start_date + timedelta(minutes=i) for i in range(num_rows)]
    
    # Generate OHLCV data
    data = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        'Volume': []
    }
    
    for i in range(num_rows):
        # Generate random price movement (can be extreme)
        price_change = draw(st.floats(min_value=-volatility, max_value=volatility, allow_nan=False, allow_infinity=False))
        open_price = max(0.01, base_price + price_change)  # Ensure positive prices
        
        # Generate high/low around open price
        high_offset = draw(st.floats(min_value=0.0, max_value=volatility, allow_nan=False, allow_infinity=False))
        low_offset = draw(st.floats(min_value=0.0, max_value=volatility, allow_nan=False, allow_infinity=False))
        
        high_price = open_price + high_offset
        low_price = max(0.01, open_price - low_offset)  # Ensure positive prices
        
        # Ensure high >= low
        if high_price < low_price:
            high_price, low_price = low_price, high_price
        
        # Close price between low and high
        close_price = draw(st.floats(min_value=low_price, max_value=high_price, allow_nan=False, allow_infinity=False))
        
        # Generate volume (can be zero or very high)
        volume_change = draw(st.integers(min_value=-base_volume, max_value=base_volume * 2))
        volume = max(0, base_volume + volume_change)
        
        data['Open'].append(open_price)
        data['High'].append(high_price)
        data['Low'].append(low_price)
        data['Close'].append(close_price)
        data['Volume'].append(volume)
    
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    return df


@given(df=ohlcv_dataframe_strategy(min_rows=10, max_rows=200))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
def test_analyze_liquidity_output_range_invariant(df):
    """
    **Property 1: Output Range Invariant**
    **Validates: Requirements 4.4**
    
    For any ticker symbol and market conditions, the anomaly_score returned by 
    analyze_liquidity MUST always be a float value within the closed interval [0.0, 1.0],
    regardless of input data outliers or extreme market conditions.
    """
    # Mock get_intraday_data to return our generated DataFrame
    with patch('src.market.get_intraday_data', return_value=df):
        result = analyze_liquidity('TEST')
    
    # Verify result is a dictionary
    assert isinstance(result, dict), "Result must be a dictionary"
    
    # Verify anomaly_score exists
    assert 'anomaly_score' in result, "Result must contain 'anomaly_score' key"
    
    # Extract anomaly score
    anomaly_score = result['anomaly_score']
    
    # Verify it's a float
    assert isinstance(anomaly_score, (float, int)), f"Anomaly score must be numeric, got {type(anomaly_score)}"
    
    # CRITICAL INVARIANT: Score must be in [0.0, 1.0]
    assert 0.0 <= anomaly_score <= 1.0, \
        f"Anomaly score {anomaly_score} is outside valid range [0.0, 1.0]"
    
    # Verify score is not NaN or infinity
    assert not np.isnan(anomaly_score), "Anomaly score must not be NaN"
    assert not np.isinf(anomaly_score), "Anomaly score must not be infinity"


@given(
    num_rows=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=50, deadline=None)
def test_analyze_liquidity_output_range_with_insufficient_data(num_rows):
    """
    Test output range invariant with insufficient data (< 10 minutes).
    
    Even with insufficient data, the function should return a neutral score (0.5)
    which is still within [0.0, 1.0].
    """
    if num_rows == 0:
        df = pd.DataFrame()
    else:
        dates = pd.date_range(start='2024-01-01 09:30', periods=num_rows, freq='1min')
        df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, num_rows),
            'High': np.random.uniform(110, 120, num_rows),
            'Low': np.random.uniform(90, 100, num_rows),
            'Close': np.random.uniform(100, 110, num_rows),
            'Volume': np.random.randint(1000000, 5000000, num_rows)
        }, index=dates)
    
    with patch('src.market.get_intraday_data', return_value=df):
        result = analyze_liquidity('TEST')
    
    anomaly_score = result['anomaly_score']
    
    # Even with insufficient data, score must be in valid range
    assert 0.0 <= anomaly_score <= 1.0, \
        f"Anomaly score {anomaly_score} is outside valid range [0.0, 1.0] even with insufficient data"
    
    # Should return neutral score
    assert anomaly_score == 0.5, "Insufficient data should return neutral score of 0.5"


@given(
    num_rows=st.integers(min_value=10, max_value=200),
    constant_price=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
    constant_volume=st.integers(min_value=0, max_value=10000000)
)
@settings(max_examples=50, deadline=None)
def test_analyze_liquidity_output_range_with_constant_prices(num_rows, constant_price, constant_volume):
    """
    Test output range invariant with constant prices (zero price range).
    
    This is an edge case where all prices are identical, which could cause
    division by zero or other numerical issues in heatmap generation.
    """
    dates = pd.date_range(start='2024-01-01 09:30', periods=num_rows, freq='1min')
    df = pd.DataFrame({
        'Open': [constant_price] * num_rows,
        'High': [constant_price] * num_rows,
        'Low': [constant_price] * num_rows,
        'Close': [constant_price] * num_rows,
        'Volume': [constant_volume] * num_rows
    }, index=dates)
    
    with patch('src.market.get_intraday_data', return_value=df):
        result = analyze_liquidity('TEST')
    
    anomaly_score = result['anomaly_score']
    
    # Even with constant prices, score must be in valid range
    assert 0.0 <= anomaly_score <= 1.0, \
        f"Anomaly score {anomaly_score} is outside valid range [0.0, 1.0] with constant prices"


@given(
    num_rows=st.integers(min_value=10, max_value=200),
)
@settings(max_examples=50, deadline=None)
def test_analyze_liquidity_output_range_with_zero_volume(num_rows):
    """
    Test output range invariant with zero volume throughout.
    
    Zero volume is a valid market condition (e.g., after-hours trading)
    and should not cause the function to return invalid scores.
    """
    dates = pd.date_range(start='2024-01-01 09:30', periods=num_rows, freq='1min')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 110, num_rows),
        'High': np.random.uniform(110, 120, num_rows),
        'Low': np.random.uniform(90, 100, num_rows),
        'Close': np.random.uniform(100, 110, num_rows),
        'Volume': [0] * num_rows  # All zero volume
    }, index=dates)
    
    with patch('src.market.get_intraday_data', return_value=df):
        result = analyze_liquidity('TEST')
    
    anomaly_score = result['anomaly_score']
    
    # Even with zero volume, score must be in valid range
    assert 0.0 <= anomaly_score <= 1.0, \
        f"Anomaly score {anomaly_score} is outside valid range [0.0, 1.0] with zero volume"


def test_analyze_liquidity_output_range_with_api_failure():
    """
    Test output range invariant when API fails.
    
    When get_intraday_data fails, the function should still return
    a valid score (neutral 0.5) within the valid range.
    """
    with patch('src.market.get_intraday_data', side_effect=Exception('API failure')):
        result = analyze_liquidity('TEST')
    
    anomaly_score = result['anomaly_score']
    
    # Even with API failure, score must be in valid range
    assert 0.0 <= anomaly_score <= 1.0, \
        f"Anomaly score {anomaly_score} is outside valid range [0.0, 1.0] with API failure"
    
    # Should return neutral score
    assert anomaly_score == 0.5, "API failure should return neutral score of 0.5"


def test_analyze_liquidity_output_range_with_heatmap_failure():
    """
    Test output range invariant when heatmap generation fails.
    
    When df_to_heatmap returns None, the function should still return
    a valid score (neutral 0.5) within the valid range.
    """
    dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='1min')
    mock_df = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(110, 120, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure.df_to_heatmap', return_value=None):
            result = analyze_liquidity('TEST')
    
    anomaly_score = result['anomaly_score']
    
    # Even with heatmap failure, score must be in valid range
    assert 0.0 <= anomaly_score <= 1.0, \
        f"Anomaly score {anomaly_score} is outside valid range [0.0, 1.0] with heatmap failure"
    
    # Should return neutral score
    assert anomaly_score == 0.5, "Heatmap failure should return neutral score of 0.5"
