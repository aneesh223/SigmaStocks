"""
Property-based tests for intraday data caching functionality.

Feature: convolutional-order-book
"""

import pytest
from hypothesis import given, strategies as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache

# Import the function to test
from src.market import get_intraday_data


@given(ticker=st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))))
def test_cache_hit_behavior(ticker):
    """
    Feature: convolutional-order-book, Property 14: Cache Hit Behavior
    **Validates: Requirements 1.3, 7.2**
    
    Test that calling get_intraday_data twice with same ticker returns cached data.
    Mock yfinance to verify API is only called once.
    """
    # Clear the cache before each test to ensure clean state
    get_intraday_data.cache_clear()
    
    # Create mock data that looks like real yfinance output
    mock_data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=pd.date_range(start=datetime.now() - timedelta(days=5), periods=100, freq='1min'))
    
    # Mock yfinance Ticker
    with patch('src.market.yf.Ticker') as mock_ticker_class:
        # Create a mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker_instance
        
        # First call - should hit the API
        result1 = get_intraday_data(ticker)
        
        # Second call with same ticker - should use cache
        result2 = get_intraday_data(ticker)
        
        # Verify the API was only called once (cache hit on second call)
        assert mock_ticker_class.call_count == 1, \
            f"Expected yf.Ticker to be called once, but was called {mock_ticker_class.call_count} times"
        
        # Verify both results are the same (cached)
        assert result1 is result2, \
            "Second call should return the exact same cached object"
        
        # Verify the data is not empty
        assert not result1.empty, "Cached data should not be empty"
        
        # Verify the data has the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            assert col in result1.columns, f"Expected column {col} in cached data"


def test_cache_hit_behavior_with_specific_ticker():
    """
    Unit test variant: Test cache behavior with a specific ticker.
    **Validates: Requirements 1.3, 7.2**
    """
    # Clear the cache before test
    get_intraday_data.cache_clear()
    
    ticker = "AAPL"
    
    # Create mock data
    mock_data = pd.DataFrame({
        'Open': [150.0, 151.0, 152.0],
        'High': [151.0, 152.0, 153.0],
        'Low': [149.0, 150.0, 151.0],
        'Close': [150.5, 151.5, 152.5],
        'Volume': [1000000, 1100000, 1200000]
    }, index=pd.date_range(start=datetime.now() - timedelta(days=1), periods=3, freq='1min'))
    
    with patch('src.market.yf.Ticker') as mock_ticker_class:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker_instance
        
        # First call
        result1 = get_intraday_data(ticker)
        
        # Second call
        result2 = get_intraday_data(ticker)
        
        # Third call with same ticker
        result3 = get_intraday_data(ticker)
        
        # Verify API was only called once despite 3 calls
        assert mock_ticker_class.call_count == 1, \
            f"Expected API to be called once, but was called {mock_ticker_class.call_count} times"
        
        # Verify all results are identical (same cached object)
        assert result1 is result2 is result3, \
            "All calls should return the same cached object"


def test_cache_different_tickers():
    """
    Unit test: Verify cache stores different data for different tickers.
    **Validates: Requirements 7.1**
    """
    # Clear the cache before test
    get_intraday_data.cache_clear()
    
    ticker1 = "AAPL"
    ticker2 = "GOOGL"
    
    # Create different mock data for each ticker
    mock_data1 = pd.DataFrame({
        'Open': [150.0],
        'High': [151.0],
        'Low': [149.0],
        'Close': [150.5],
        'Volume': [1000000]
    }, index=pd.date_range(start=datetime.now(), periods=1, freq='1min'))
    
    mock_data2 = pd.DataFrame({
        'Open': [2800.0],
        'High': [2850.0],
        'Low': [2780.0],
        'Close': [2820.0],
        'Volume': [500000]
    }, index=pd.date_range(start=datetime.now(), periods=1, freq='1min'))
    
    with patch('src.market.yf.Ticker') as mock_ticker_class:
        def mock_ticker_side_effect(ticker):
            mock_instance = MagicMock()
            if ticker == ticker1:
                mock_instance.history.return_value = mock_data1
            else:
                mock_instance.history.return_value = mock_data2
            return mock_instance
        
        mock_ticker_class.side_effect = mock_ticker_side_effect
        
        # Call with first ticker
        result1 = get_intraday_data(ticker1)
        
        # Call with second ticker
        result2 = get_intraday_data(ticker2)
        
        # Call with first ticker again (should be cached)
        result1_cached = get_intraday_data(ticker1)
        
        # Verify API was called twice (once per unique ticker)
        assert mock_ticker_class.call_count == 2, \
            f"Expected API to be called twice (once per ticker), but was called {mock_ticker_class.call_count} times"
        
        # Verify first ticker's cached result is the same object
        assert result1 is result1_cached, \
            "Cached result for first ticker should be the same object"
        
        # Verify the data is different for different tickers
        assert not result1.equals(result2), \
            "Different tickers should return different data"


def test_cache_with_empty_data():
    """
    Unit test: Verify cache behavior when API returns empty data.
    **Validates: Requirements 1.5**
    """
    # Clear the cache before test
    get_intraday_data.cache_clear()
    
    ticker = "INVALID"
    
    # Mock empty DataFrame
    mock_empty_data = pd.DataFrame()
    
    with patch('src.market.yf.Ticker') as mock_ticker_class:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_empty_data
        mock_ticker_class.return_value = mock_ticker_instance
        
        # First call
        result1 = get_intraday_data(ticker)
        
        # Second call
        result2 = get_intraday_data(ticker)
        
        # Verify API was only called once (empty data is also cached)
        assert mock_ticker_class.call_count == 1, \
            f"Expected API to be called once, but was called {mock_ticker_class.call_count} times"
        
        # Verify both results are empty
        assert result1.empty, "First result should be empty"
        assert result2.empty, "Second result should be empty"
        
        # Verify cached result is the same object
        assert result1 is result2, \
            "Cached empty result should be the same object"
