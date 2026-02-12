"""
Property-based tests for heatmap shape invariant.

Feature: convolutional-order-book
Task: 5.2 Write property test for heatmap shape
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from src.microstructure import df_to_heatmap


# Custom strategy for generating OHLCV DataFrames
@st.composite
def ohlcv_dataframe(draw, min_rows=1, max_rows=500):
    """
    Generate random OHLCV DataFrame with varying lengths.
    
    Args:
        draw: Hypothesis draw function
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows
    
    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume]
    """
    # Generate random number of rows
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    # Generate base price (avoid extreme values that might cause numerical issues)
    base_price = draw(st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    
    # Generate price volatility (how much prices vary)
    volatility = draw(st.floats(min_value=0.0, max_value=base_price * 0.5, allow_nan=False, allow_infinity=False))
    
    # Generate base volume
    base_volume = draw(st.integers(min_value=0, max_value=10000000))
    
    # Create timestamps
    start_time = datetime.now() - timedelta(minutes=num_rows)
    dates = pd.date_range(start=start_time, periods=num_rows, freq='1min')
    
    # Generate OHLCV data
    data = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        'Volume': []
    }
    
    for i in range(num_rows):
        # Generate random price movement
        price_change = draw(st.floats(min_value=-volatility, max_value=volatility, allow_nan=False, allow_infinity=False))
        open_price = base_price + price_change
        
        # Generate high/low around open price
        high_offset = draw(st.floats(min_value=0.0, max_value=volatility * 0.5, allow_nan=False, allow_infinity=False))
        low_offset = draw(st.floats(min_value=0.0, max_value=volatility * 0.5, allow_nan=False, allow_infinity=False))
        
        high_price = open_price + high_offset
        low_price = max(0.01, open_price - low_offset)  # Ensure positive price
        
        # Close price between low and high
        close_price = draw(st.floats(min_value=low_price, max_value=high_price, allow_nan=False, allow_infinity=False))
        
        # Generate volume
        volume_change = draw(st.integers(min_value=-base_volume // 2, max_value=base_volume))
        volume = max(0, base_volume + volume_change)
        
        data['Open'].append(open_price)
        data['High'].append(high_price)
        data['Low'].append(low_price)
        data['Close'].append(close_price)
        data['Volume'].append(volume)
    
    return pd.DataFrame(data, index=dates)


@given(df=ohlcv_dataframe(min_rows=1, max_rows=500))
@settings(suppress_health_check=[HealthCheck.data_too_large])
def test_heatmap_shape_invariant(df):
    """
    Feature: convolutional-order-book, Property 3: Dimensional Consistency
    **Validates: Requirements 2.1, 2.6**
    
    Test that for any input DataFrame, the df_to_heatmap function always returns
    a numpy array of shape (64, 64) or None. It must handle input DataFrames with
    length < 64 by padding with zeros, ensuring the CNN input layer never receives
    mismatched dimensions.
    
    This property must hold for:
    - Any DataFrame length from 1 to 500 rows
    - Any price range (including zero price range)
    - Any volume values (including zero volume)
    - Various price volatilities
    """
    heatmap = df_to_heatmap(df)
    
    # The function should either return a valid (64, 64) heatmap or None
    if heatmap is not None:
        assert heatmap.shape == (64, 64), \
            f"Expected shape (64, 64), but got {heatmap.shape} for DataFrame with {len(df)} rows"
        
        # Verify dtype is float32
        assert heatmap.dtype == np.float32, \
            f"Expected dtype float32, but got {heatmap.dtype}"
        
        # Verify no NaN or Inf values in output
        assert not np.any(np.isnan(heatmap)), \
            "Heatmap should not contain NaN values"
        
        assert not np.any(np.isinf(heatmap)), \
            "Heatmap should not contain Inf values"


@given(
    num_rows=st.integers(min_value=1, max_value=63),
    base_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    base_volume=st.integers(min_value=0, max_value=1000000)
)
def test_heatmap_shape_with_insufficient_data(num_rows, base_price, base_volume):
    """
    Feature: convolutional-order-book, Property 3: Dimensional Consistency
    **Validates: Requirements 2.1, 2.6**
    
    Test that heatmap has shape (64, 64) even when given fewer than 64 rows.
    The function should pad with zeros to maintain consistent dimensions.
    """
    # Create DataFrame with fewer than 64 rows
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [base_price] * num_rows,
        'High': [base_price + 1.0] * num_rows,
        'Low': [base_price - 1.0] * num_rows,
        'Close': [base_price + 0.5] * num_rows,
        'Volume': [base_volume] * num_rows
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, \
        f"Heatmap should not be None for {num_rows} rows of data"
    
    assert heatmap.shape == (64, 64), \
        f"Expected shape (64, 64) even with {num_rows} rows, but got {heatmap.shape}"


@given(
    num_rows=st.integers(min_value=65, max_value=500),
    base_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    base_volume=st.integers(min_value=0, max_value=1000000)
)
def test_heatmap_shape_with_excess_data(num_rows, base_price, base_volume):
    """
    Feature: convolutional-order-book, Property 3: Dimensional Consistency
    **Validates: Requirements 2.1, 2.6**
    
    Test that heatmap has shape (64, 64) even when given more than 64 rows.
    The function should use only the last 64 rows.
    """
    # Create DataFrame with more than 64 rows
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [base_price + i * 0.1 for i in range(num_rows)],
        'High': [base_price + i * 0.1 + 1.0 for i in range(num_rows)],
        'Low': [base_price + i * 0.1 - 1.0 for i in range(num_rows)],
        'Close': [base_price + i * 0.1 + 0.5 for i in range(num_rows)],
        'Volume': [base_volume + i * 100 for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, \
        f"Heatmap should not be None for {num_rows} rows of data"
    
    assert heatmap.shape == (64, 64), \
        f"Expected shape (64, 64) even with {num_rows} rows, but got {heatmap.shape}"


@given(
    num_rows=st.integers(min_value=1, max_value=200),
    window_size=st.integers(min_value=16, max_value=128)
)
def test_heatmap_shape_with_custom_window_size(num_rows, window_size):
    """
    Feature: convolutional-order-book, Property 3: Dimensional Consistency
    **Validates: Requirements 2.1**
    
    Test that heatmap respects custom window_size parameter and always returns
    the specified shape.
    """
    # Create DataFrame
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [100.0] * num_rows,
        'High': [101.0] * num_rows,
        'Low': [99.0] * num_rows,
        'Close': [100.5] * num_rows,
        'Volume': [1000000] * num_rows
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df, window_size=window_size)
    
    if heatmap is not None:
        assert heatmap.shape == (window_size, window_size), \
            f"Expected shape ({window_size}, {window_size}), but got {heatmap.shape}"


@given(
    num_rows=st.integers(min_value=10, max_value=100),
    price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    volume=st.integers(min_value=0, max_value=10000000)
)
def test_heatmap_shape_with_zero_price_range(num_rows, price, volume):
    """
    Feature: convolutional-order-book, Property 3: Dimensional Consistency
    **Validates: Requirements 2.1, 2.7**
    
    Test that heatmap has shape (64, 64) even when all prices are identical
    (zero price range). The function should return a uniform heatmap.
    """
    # Create DataFrame with zero price range (all prices identical)
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [price] * num_rows,
        'High': [price] * num_rows,
        'Low': [price] * num_rows,
        'Close': [price] * num_rows,
        'Volume': [volume] * num_rows
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, \
        "Heatmap should not be None even with zero price range"
    
    assert heatmap.shape == (64, 64), \
        f"Expected shape (64, 64) with zero price range, but got {heatmap.shape}"


def test_heatmap_shape_with_empty_dataframe():
    """
    Unit test: Verify that empty DataFrame returns None.
    **Validates: Requirements 2.6**
    """
    df = pd.DataFrame()
    heatmap = df_to_heatmap(df)
    
    assert heatmap is None, \
        "Heatmap should be None for empty DataFrame"


def test_heatmap_shape_with_missing_columns():
    """
    Unit test: Verify that DataFrame with missing columns returns None.
    """
    dates = pd.date_range(start=datetime.now(), periods=64, freq='1min')
    
    # Missing 'Volume' column
    data = {
        'Open': [100.0] * 64,
        'High': [101.0] * 64,
        'Low': [99.0] * 64,
        'Close': [100.5] * 64
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is None, \
        "Heatmap should be None when required columns are missing"


@given(
    num_rows=st.integers(min_value=10, max_value=100),
    nan_count=st.integers(min_value=1, max_value=10)
)
def test_heatmap_shape_with_nan_values(num_rows, nan_count):
    """
    Feature: convolutional-order-book, Property 3: Dimensional Consistency
    **Validates: Requirements 2.1**
    
    Test that heatmap has shape (64, 64) even when DataFrame contains NaN values.
    The function should handle NaN values gracefully.
    """
    # Create DataFrame with some NaN values
    dates = pd.date_range(start=datetime.now(), periods=num_rows, freq='1min')
    
    data = {
        'Open': [100.0 + i * 0.1 for i in range(num_rows)],
        'High': [101.0 + i * 0.1 for i in range(num_rows)],
        'Low': [99.0 + i * 0.1 for i in range(num_rows)],
        'Close': [100.5 + i * 0.1 for i in range(num_rows)],
        'Volume': [1000000 + i * 1000 for i in range(num_rows)]
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Introduce NaN values at random positions
    for _ in range(min(nan_count, num_rows)):
        idx = np.random.randint(0, num_rows)
        col = np.random.choice(['High', 'Low', 'Volume'])
        df.iloc[idx, df.columns.get_loc(col)] = np.nan
    
    heatmap = df_to_heatmap(df)
    
    if heatmap is not None:
        assert heatmap.shape == (64, 64), \
            f"Expected shape (64, 64) with NaN values, but got {heatmap.shape}"
        
        # Verify output doesn't contain NaN
        assert not np.any(np.isnan(heatmap)), \
            "Heatmap should not contain NaN values even if input has NaN"
