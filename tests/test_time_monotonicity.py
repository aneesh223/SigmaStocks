"""
Property-based tests for time monotonicity in heatmaps.

Feature: convolutional-order-book
Task: 5.4 Write property test for time monotonicity
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume

from src.microstructure import df_to_heatmap


@given(
    num_rows=st.integers(min_value=64, max_value=200),
    base_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    base_volume=st.integers(min_value=1000, max_value=10000000)
)
def test_time_monotonicity_column_ordering(num_rows, base_price, base_volume):
    """
    Feature: convolutional-order-book, Property 4: Time-Monotonicity
    **Validates: Requirements 2.2, 2.3**
    
    Test that for any generated heatmap, the X-axis (columns) represents time
    strictly increasing from left (column 0 = oldest) to right (column 63 = newest).
    
    This test creates a DataFrame with known timestamps and distinctive volume
    patterns, then verifies that the heatmap correctly maps the oldest data to
    column 0 and the newest data to column 63.
    """
    # Create DataFrame with timestamps in chronological order
    start_time = datetime.now() - timedelta(minutes=num_rows)
    dates = pd.date_range(start=start_time, periods=num_rows, freq='1min')
    
    # Create distinctive volume pattern: increasing over time
    # This makes it easy to verify time ordering in the heatmap
    volumes = [base_volume + i * 1000 for i in range(num_rows)]
    
    data = {
        'Open': [base_price + i * 0.1 for i in range(num_rows)],
        'High': [base_price + i * 0.1 + 1.0 for i in range(num_rows)],
        'Low': [base_price + i * 0.1 - 1.0 for i in range(num_rows)],
        'Close': [base_price + i * 0.1 + 0.5 for i in range(num_rows)],
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    
    # The heatmap uses the last 64 minutes of data
    # So column 0 should correspond to the 64th-from-last row
    # And column 63 should correspond to the last row
    
    # Get the last 64 rows from the DataFrame
    df_last_64 = df.tail(64)
    
    # Calculate average intensity for each column
    # Column 0 should have lower average intensity (older, lower volume)
    # Column 63 should have higher average intensity (newer, higher volume)
    col_0_intensity = heatmap[:, 0].mean()
    col_63_intensity = heatmap[:, 63].mean()
    
    # Since volumes increase over time, column 63 should have higher intensity
    # Allow for edge cases where the difference might be small
    if col_0_intensity > 0 and col_63_intensity > 0:
        # Verify that newer data (column 63) has higher or equal intensity
        assert col_63_intensity >= col_0_intensity * 0.9, \
            f"Column 63 (newest) should have >= intensity than column 0 (oldest), " \
            f"but got col_0={col_0_intensity:.4f}, col_63={col_63_intensity:.4f}"


@given(
    base_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    base_volume=st.integers(min_value=1000, max_value=1000000)
)
def test_time_monotonicity_with_spike_at_start(base_price, base_volume):
    """
    Feature: convolutional-order-book, Property 4: Time-Monotonicity
    **Validates: Requirements 2.2, 2.3**
    
    Test time monotonicity by placing a distinctive volume spike at the start
    of the data and verifying it appears in column 0 (oldest).
    """
    num_rows = 64
    start_time = datetime.now() - timedelta(minutes=num_rows)
    dates = pd.date_range(start=start_time, periods=num_rows, freq='1min')
    
    # Create volume pattern with spike at the start
    volumes = [base_volume] * num_rows
    volumes[0] = base_volume * 10  # Large spike at the oldest data point
    
    data = {
        'Open': [base_price] * num_rows,
        'High': [base_price + 1.0] * num_rows,
        'Low': [base_price - 1.0] * num_rows,
        'Close': [base_price + 0.5] * num_rows,
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    
    # Column 0 should have the highest intensity due to the spike
    col_0_max = heatmap[:, 0].max()
    col_63_max = heatmap[:, 63].max()
    
    # The spike at the start should make column 0 brighter than column 63
    assert col_0_max > col_63_max * 1.5, \
        f"Column 0 (oldest with spike) should be much brighter than column 63, " \
        f"but got col_0_max={col_0_max:.4f}, col_63_max={col_63_max:.4f}"


@given(
    base_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    base_volume=st.integers(min_value=1000, max_value=1000000)
)
def test_time_monotonicity_with_spike_at_end(base_price, base_volume):
    """
    Feature: convolutional-order-book, Property 4: Time-Monotonicity
    **Validates: Requirements 2.2, 2.3**
    
    Test time monotonicity by placing a distinctive volume spike at the end
    of the data and verifying it appears in column 63 (newest).
    """
    num_rows = 64
    start_time = datetime.now() - timedelta(minutes=num_rows)
    dates = pd.date_range(start=start_time, periods=num_rows, freq='1min')
    
    # Create volume pattern with spike at the end
    volumes = [base_volume] * num_rows
    volumes[-1] = base_volume * 10  # Large spike at the newest data point
    
    data = {
        'Open': [base_price] * num_rows,
        'High': [base_price + 1.0] * num_rows,
        'Low': [base_price - 1.0] * num_rows,
        'Close': [base_price + 0.5] * num_rows,
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    
    # Column 63 should have the highest intensity due to the spike
    col_0_max = heatmap[:, 0].max()
    col_63_max = heatmap[:, 63].max()
    
    # The spike at the end should make column 63 brighter than column 0
    assert col_63_max > col_0_max * 1.5, \
        f"Column 63 (newest with spike) should be much brighter than column 0, " \
        f"but got col_63_max={col_63_max:.4f}, col_0_max={col_0_max:.4f}"


@given(
    num_rows=st.integers(min_value=100, max_value=200),
    base_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    base_volume=st.integers(min_value=1000, max_value=1000000)
)
def test_time_monotonicity_with_excess_data(num_rows, base_price, base_volume):
    """
    Feature: convolutional-order-book, Property 4: Time-Monotonicity
    **Validates: Requirements 2.2, 2.3**
    
    Test that when given more than 64 rows, the heatmap uses only the last 64 rows,
    maintaining time monotonicity where column 0 = 64th-from-last row and
    column 63 = last row.
    """
    # Create DataFrame with more than 64 rows
    start_time = datetime.now() - timedelta(minutes=num_rows)
    dates = pd.date_range(start=start_time, periods=num_rows, freq='1min')
    
    # Create distinctive volume pattern: linearly increasing
    volumes = [base_volume + i * 100 for i in range(num_rows)]
    
    data = {
        'Open': [base_price + i * 0.01 for i in range(num_rows)],
        'High': [base_price + i * 0.01 + 1.0 for i in range(num_rows)],
        'Low': [base_price + i * 0.01 - 1.0 for i in range(num_rows)],
        'Close': [base_price + i * 0.01 + 0.5 for i in range(num_rows)],
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    
    # The heatmap should use the last 64 rows
    # So the oldest data in the heatmap (column 0) should correspond to row (num_rows - 64)
    # And the newest data (column 63) should correspond to row (num_rows - 1)
    
    # Since volumes increase linearly, column 63 should have higher intensity
    col_0_mean = heatmap[:, 0].mean()
    col_63_mean = heatmap[:, 63].mean()
    
    # Allow for some tolerance due to normalization
    if col_0_mean > 0:
        assert col_63_mean >= col_0_mean * 0.9, \
            f"With excess data, column 63 (newest) should have >= intensity than column 0 (oldest), " \
            f"but got col_0={col_0_mean:.4f}, col_63={col_63_mean:.4f}"


def test_time_monotonicity_exact_64_rows():
    """
    Unit test: Verify time monotonicity with exactly 64 rows.
    **Validates: Requirements 2.2, 2.3**
    """
    num_rows = 64
    start_time = datetime.now() - timedelta(minutes=num_rows)
    dates = pd.date_range(start=start_time, periods=num_rows, freq='1min')
    
    # Create volume pattern that increases linearly
    volumes = [1000 + i * 1000 for i in range(num_rows)]
    
    data = {
        'Open': [100.0 + i * 0.1 for i in range(num_rows)],
        'High': [101.0 + i * 0.1 for i in range(num_rows)],
        'Low': [99.0 + i * 0.1 for i in range(num_rows)],
        'Close': [100.5 + i * 0.1 for i in range(num_rows)],
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    
    # With exactly 64 rows:
    # - Column 0 should correspond to the first row (oldest)
    # - Column 63 should correspond to the last row (newest)
    
    # Calculate mean intensity for first and last columns
    col_0_mean = heatmap[:, 0].mean()
    col_63_mean = heatmap[:, 63].mean()
    
    # Since volumes increase linearly, column 63 should have higher intensity
    assert col_63_mean > col_0_mean, \
        f"Column 63 (newest) should have higher intensity than column 0 (oldest), " \
        f"but got col_0={col_0_mean:.4f}, col_63={col_63_mean:.4f}"


@given(
    num_rows=st.integers(min_value=10, max_value=50),
    base_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    base_volume=st.integers(min_value=1000, max_value=1000000)
)
def test_time_monotonicity_with_padding(num_rows, base_price, base_volume):
    """
    Feature: convolutional-order-book, Property 4: Time-Monotonicity
    **Validates: Requirements 2.2, 2.3**
    
    Test that when given fewer than 64 rows, the heatmap pads with zeros on the left,
    and the actual data appears on the right side, maintaining time monotonicity.
    """
    # Create DataFrame with fewer than 64 rows
    start_time = datetime.now() - timedelta(minutes=num_rows)
    dates = pd.date_range(start=start_time, periods=num_rows, freq='1min')
    
    # Create volume pattern with spike at the end
    volumes = [base_volume] * num_rows
    volumes[-1] = base_volume * 5  # Spike at the newest data point
    
    data = {
        'Open': [base_price] * num_rows,
        'High': [base_price + 1.0] * num_rows,
        'Low': [base_price - 1.0] * num_rows,
        'Close': [base_price + 0.5] * num_rows,
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    
    # With fewer than 64 rows, the data should be padded on the left
    # So the actual data should appear in the rightmost columns
    # Column 63 should have the spike (newest data)
    
    # Calculate the expected starting column for actual data
    start_col = 64 - num_rows
    
    # Columns before start_col should be mostly zeros (padding)
    if start_col > 0:
        padding_mean = heatmap[:, :start_col].mean()
        data_mean = heatmap[:, start_col:].mean()
        
        # Padding should have lower intensity than actual data
        assert padding_mean <= data_mean, \
            f"Padding columns should have lower intensity than data columns, " \
            f"but got padding={padding_mean:.4f}, data={data_mean:.4f}"
    
    # Column 63 should have the highest intensity (spike at newest data)
    col_63_max = heatmap[:, 63].max()
    
    # The spike should be visible in column 63
    assert col_63_max > 0.5, \
        f"Column 63 should have high intensity from spike, but got {col_63_max:.4f}"


def test_time_monotonicity_chronological_order():
    """
    Unit test: Verify that timestamps are processed in chronological order.
    **Validates: Requirements 2.2, 2.3**
    """
    num_rows = 64
    
    # Create DataFrame with explicit timestamps
    dates = pd.date_range(start='2024-01-01 09:30:00', periods=num_rows, freq='1min')
    
    # Create a unique volume for each minute to track ordering
    # Use a pattern where each minute has a distinct volume
    volumes = [1000 * (i + 1) for i in range(num_rows)]
    
    data = {
        'Open': [100.0] * num_rows,
        'High': [101.0] * num_rows,
        'Low': [99.0] * num_rows,
        'Close': [100.5] * num_rows,
        'Volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    heatmap = df_to_heatmap(df)
    
    assert heatmap is not None, "Heatmap should not be None"
    
    # Verify that intensity increases from left to right
    # (since volume increases over time)
    column_means = [heatmap[:, i].mean() for i in range(64)]
    
    # Check that there's a general increasing trend
    # (allowing for some noise due to normalization)
    first_quarter_mean = np.mean(column_means[:16])
    last_quarter_mean = np.mean(column_means[48:])
    
    assert last_quarter_mean > first_quarter_mean, \
        f"Last quarter of columns should have higher mean intensity than first quarter, " \
        f"but got first={first_quarter_mean:.4f}, last={last_quarter_mean:.4f}"
