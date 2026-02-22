"""
Unit tests for error handling in microstructure analysis.

Feature: convolutional-order-book
Task: 10.5 Write unit tests for error handling

Tests that the system handles errors gracefully and logs appropriately.
"""

import pytest
import pandas as pd
import logging
from unittest.mock import patch, MagicMock
import torch

from src.microstructure import analyze_liquidity, df_to_heatmap, _get_model
from src.market import get_intraday_data


def test_cnn_model_load_failure_logs_error_and_continues():
    """
    Test that CNN model load failure logs error and continues.
    
    The system should handle model initialization failures gracefully
    and return a neutral score without crashing.
    
    **Validates: Requirements 6.2**
    """
    # Reset the global model to None to test initialization failure
    import src.microstructure
    original_model = src.microstructure._model
    original_device = src.microstructure._device
    src.microstructure._model = None
    src.microstructure._device = None
    
    try:
        # Mock torch.device to raise an exception
        with patch('src.microstructure.torch.device', side_effect=Exception("CUDA error")):
            with patch('src.microstructure.logger') as mock_logger:
                # Try to analyze liquidity - should handle the error
                result = analyze_liquidity('TEST')
                
                # Should return neutral score
                assert result['anomaly_score'] == 0.5
                assert result['confidence'] == 'low'
                
                # Should log error
                mock_logger.error.assert_called()
    finally:
        # Restore original model state
        src.microstructure._model = original_model
        src.microstructure._device = original_device


def test_intraday_data_fetch_failure_logs_warning_and_returns_neutral_score():
    """
    Test that intraday data fetch failure logs warning and returns neutral score.
    
    When the API fails to fetch data, the system should log a warning
    and return a neutral anomaly score.
    
    **Validates: Requirements 6.3**
    """
    # Mock get_intraday_data to return empty DataFrame
    with patch('src.market.get_intraday_data', return_value=pd.DataFrame()):
        with patch('src.microstructure.logger') as mock_logger:
            result = analyze_liquidity('TEST')
            
            # Should return neutral score
            assert result['anomaly_score'] == 0.5
            assert result['status'] == 'Insufficient data for microstructure analysis'
            assert result['confidence'] == 'low'
            
            # Should log warning
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert 'Insufficient data' in call_args


def test_heatmap_generation_failure_logs_error_and_returns_none():
    """
    Test that heatmap generation failure logs error and returns None.
    
    When heatmap generation fails due to invalid data, the system
    should log an error and return None.
    
    **Validates: Requirements 6.4**
    """
    # Create invalid DataFrame (missing required columns)
    invalid_df = pd.DataFrame({
        'Open': [100, 101, 102],
        'Close': [100, 101, 102]
        # Missing 'High', 'Low', 'Volume'
    })
    
    with patch('src.microstructure.logger') as mock_logger:
        result = df_to_heatmap(invalid_df)
        
        # Should return None
        assert result is None
        
        # Should log error
        mock_logger.error.assert_called()
        call_args = str(mock_logger.error.call_args)
        assert 'missing required columns' in call_args.lower()


def test_heatmap_generation_exception_logs_error_and_returns_none():
    """
    Test that exceptions during heatmap generation are logged and return None.
    
    **Validates: Requirements 6.4**
    """
    # Create a DataFrame that will cause an exception
    df = pd.DataFrame({
        'High': [100, 101, 102],
        'Low': [99, 100, 101],
        'Volume': [1000, 1100, 1200]
    })
    
    # Mock numpy to raise an exception
    with patch('src.microstructure.np.zeros', side_effect=Exception("Memory error")):
        with patch('src.microstructure.logger') as mock_logger:
            result = df_to_heatmap(df)
            
            # Should return None
            assert result is None
            
            # Should log error with exception details
            mock_logger.error.assert_called()
            call_args = str(mock_logger.error.call_args)
            assert 'Failed to generate heatmap' in call_args


def test_cnn_inference_failure_logs_error_and_returns_neutral_score():
    """
    Test that CNN inference failure logs error and returns neutral score.
    
    When the CNN model fails during inference, the system should
    log an error and return a neutral anomaly score.
    
    **Validates: Requirements 6.5**
    """
    # Create valid test data
    dates = pd.date_range(start='2024-01-01', periods=64, freq='1min')
    df = pd.DataFrame({
        'High': [100 + i * 0.1 for i in range(64)],
        'Low': [99 + i * 0.1 for i in range(64)],
        'Volume': [1000 + i * 10 for i in range(64)]
    }, index=dates)
    
    # Mock get_intraday_data to return valid data
    with patch('src.market.get_intraday_data', return_value=df):
        # Mock the model to raise an exception during inference
        mock_model = MagicMock()
        mock_model.side_effect = Exception("CUDA out of memory")
        
        with patch('src.microstructure._get_model', return_value=(mock_model, torch.device('cpu'))):
            with patch('src.microstructure.logger') as mock_logger:
                result = analyze_liquidity('TEST')
                
                # Should return neutral score
                assert result['anomaly_score'] == 0.5
                assert result['confidence'] == 'low'
                
                # Should log error
                mock_logger.error.assert_called()
                call_args = str(mock_logger.error.call_args)
                assert 'Microstructure analysis failed' in call_args


def test_intraday_data_api_exception_logs_error():
    """
    Test that API exceptions in get_intraday_data are logged.
    
    **Validates: Requirements 6.3**
    """
    # Clear the cache to ensure fresh call
    get_intraday_data.cache_clear()
    
    # Mock yfinance to raise an exception
    with patch('src.market.yf.Ticker') as mock_ticker:
        mock_ticker.side_effect = Exception("Network timeout")
        
        with patch('src.market.logging') as mock_logging:
            result = get_intraday_data('TEST')
            
            # Should return empty DataFrame
            assert result.empty
            
            # Should log error
            mock_logging.error.assert_called()
            call_args = str(mock_logging.error.call_args)
            assert 'Error fetching intraday data' in call_args
            assert 'Network timeout' in call_args


def test_analyze_liquidity_handles_all_exceptions():
    """
    Test that analyze_liquidity handles all types of exceptions gracefully.
    
    The function should never crash, regardless of what goes wrong.
    
    **Validates: Requirements 6.2, 6.5**
    """
    # Mock get_intraday_data to raise an unexpected exception
    with patch('src.market.get_intraday_data', side_effect=RuntimeError("Unexpected error")):
        with patch('src.microstructure.logger') as mock_logger:
            result = analyze_liquidity('TEST')
            
            # Should return neutral score
            assert result['anomaly_score'] == 0.5
            assert result['confidence'] == 'low'
            assert 'Insufficient data' in result['status']
            
            # Should log error
            mock_logger.error.assert_called()


def test_empty_dataframe_logs_warning():
    """
    Test that empty DataFrame in get_intraday_data logs warning.
    
    **Validates: Requirements 6.3**
    """
    # Clear the cache
    get_intraday_data.cache_clear()
    
    # Mock yfinance to return empty DataFrame
    with patch('src.market.yf.Ticker') as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance
        
        with patch('src.market.logging') as mock_logging:
            result = get_intraday_data('TEST')
            
            # Should return empty DataFrame
            assert result.empty
            
            # Should log warning
            mock_logging.warning.assert_called()
            call_args = str(mock_logging.warning.call_args)
            assert 'No intraday data found' in call_args


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
