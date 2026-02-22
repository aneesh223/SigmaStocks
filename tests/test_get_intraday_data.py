"""
Unit tests for get_intraday_data function.

Feature: convolutional-order-book
Task: 2.3 Write unit tests for get_intraday_data
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import the function to test
from src.market import get_intraday_data


def test_successful_data_fetch_returns_dataframe_with_correct_columns():
    """
    Test successful data fetch returns DataFrame with correct columns.
    **Validates: Requirements 1.1**
    """
    # Clear the cache before test
    get_intraday_data.cache_clear()
    
    ticker = "AAPL"
    
    # Create mock data with the expected columns
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
        
        # Call the function
        result = get_intraday_data(ticker)
        
        # Verify the result is a DataFrame
        assert isinstance(result, pd.DataFrame), \
            "Result should be a pandas DataFrame"
        
        # Verify the DataFrame is not empty
        assert not result.empty, \
            "Result DataFrame should not be empty"
        
        # Verify the DataFrame has the correct columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            assert col in result.columns, \
                f"Expected column '{col}' in result DataFrame"
        
        # Verify the API was called with correct parameters
        mock_ticker_instance.history.assert_called_once_with(period="5d", interval="1m")
        
        # Verify the data values are correct
        assert len(result) == 3, "Result should have 3 rows"
        assert result['Close'].iloc[0] == 150.5, "First close price should be 150.5"


def test_empty_api_response_returns_empty_dataframe():
    """
    Test empty API response returns empty DataFrame.
    **Validates: Requirements 1.5**
    """
    # Clear the cache before test
    get_intraday_data.cache_clear()
    
    ticker = "INVALID"
    
    # Mock empty DataFrame (simulating no data available)
    mock_empty_data = pd.DataFrame()
    
    with patch('src.market.yf.Ticker') as mock_ticker_class:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_empty_data
        mock_ticker_class.return_value = mock_ticker_instance
        
        # Call the function
        result = get_intraday_data(ticker)
        
        # Verify the result is a DataFrame
        assert isinstance(result, pd.DataFrame), \
            "Result should be a pandas DataFrame even when empty"
        
        # Verify the DataFrame is empty
        assert result.empty, \
            "Result DataFrame should be empty when API returns no data"
        
        # Verify the API was called
        mock_ticker_instance.history.assert_called_once_with(period="5d", interval="1m")


def test_function_logs_warning_on_empty_data():
    """
    Test function logs warning on empty data.
    **Validates: Requirements 1.5**
    """
    # Clear the cache before test
    get_intraday_data.cache_clear()
    
    ticker = "NODATA"
    
    # Mock empty DataFrame
    mock_empty_data = pd.DataFrame()
    
    with patch('src.market.yf.Ticker') as mock_ticker_class, \
         patch('src.market.logging.warning') as mock_warning:
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_empty_data
        mock_ticker_class.return_value = mock_ticker_instance
        
        # Call the function
        result = get_intraday_data(ticker)
        
        # Verify the result is empty
        assert result.empty, "Result should be empty"
        
        # Verify a warning was logged
        mock_warning.assert_called_once()
        
        # Verify the warning message contains the ticker
        warning_call_args = mock_warning.call_args[0][0]
        assert ticker in warning_call_args, \
            f"Warning message should contain ticker '{ticker}'"
        assert "No intraday data found" in warning_call_args, \
            "Warning message should indicate no data was found"


def test_api_exception_returns_empty_dataframe_and_logs_warning():
    """
    Test that API exceptions are handled gracefully.
    **Validates: Requirements 1.5**
    """
    # Clear the cache before test
    get_intraday_data.cache_clear()
    
    ticker = "ERROR"
    
    with patch('src.market.yf.Ticker') as mock_ticker_class, \
         patch('src.market.logging.warning') as mock_warning:
        
        # Mock an exception during API call
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("Network error")
        mock_ticker_class.return_value = mock_ticker_instance
        
        # Call the function
        result = get_intraday_data(ticker)
        
        # Verify the result is an empty DataFrame
        assert isinstance(result, pd.DataFrame), \
            "Result should be a DataFrame even on exception"
        assert result.empty, \
            "Result should be empty when exception occurs"
        
        # Verify a warning was logged
        mock_warning.assert_called_once()
        
        # Verify the warning message contains error information
        warning_call_args = mock_warning.call_args[0][0]
        assert ticker in warning_call_args, \
            f"Warning message should contain ticker '{ticker}'"
        assert "Error fetching intraday data" in warning_call_args, \
            "Warning message should indicate an error occurred"


def test_dataframe_has_datetime_index():
    """
    Test that the returned DataFrame has a DatetimeIndex.
    **Validates: Requirements 1.1**
    """
    # Clear the cache before test
    get_intraday_data.cache_clear()
    
    ticker = "MSFT"
    
    # Create mock data with DatetimeIndex
    date_index = pd.date_range(start=datetime.now() - timedelta(days=2), periods=10, freq='1min')
    mock_data = pd.DataFrame({
        'Open': np.random.uniform(300, 310, 10),
        'High': np.random.uniform(300, 310, 10),
        'Low': np.random.uniform(300, 310, 10),
        'Close': np.random.uniform(300, 310, 10),
        'Volume': np.random.randint(500000, 1000000, 10)
    }, index=date_index)
    
    with patch('src.market.yf.Ticker') as mock_ticker_class:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker_instance
        
        # Call the function
        result = get_intraday_data(ticker)
        
        # Verify the index is a DatetimeIndex
        assert isinstance(result.index, pd.DatetimeIndex), \
            "Result DataFrame should have a DatetimeIndex"
        
        # Verify the index has the correct number of entries
        assert len(result.index) == 10, \
            "Result should have 10 time entries"
