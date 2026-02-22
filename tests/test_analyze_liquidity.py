"""
Unit tests for analyze_liquidity function.

Tests the microstructure analysis function that integrates data fetching,
heatmap generation, and CNN inference.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.microstructure import analyze_liquidity


class TestAnalyzeLiquidity:
    """Test suite for analyze_liquidity function."""
    
    def test_analyze_liquidity_with_sufficient_data(self):
        """Test analyze_liquidity returns valid result with sufficient data."""
        # Create mock DataFrame with 100 minutes of data
        dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='1min')
        mock_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        with patch('src.market.get_intraday_data', return_value=mock_df):
            result = analyze_liquidity('AAPL')
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'anomaly_score' in result
        assert 'status' in result
        assert 'confidence' in result
        
        # Verify anomaly_score is in valid range
        assert 0.0 <= result['anomaly_score'] <= 1.0
        
        # Verify status is one of the expected messages
        valid_statuses = [
            'Critical liquidity void detected',
            'High microstructure volatility',
            'Normal market conditions',
            'Strong accumulation zone detected',
            'Insufficient data for microstructure analysis'
        ]
        assert result['status'] in valid_statuses
    
    def test_analyze_liquidity_with_insufficient_data(self):
        """Test analyze_liquidity returns neutral score with insufficient data."""
        # Create mock DataFrame with only 5 minutes of data
        dates = pd.date_range(start='2024-01-01 09:30', periods=5, freq='1min')
        mock_df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        with patch('src.market.get_intraday_data', return_value=mock_df):
            result = analyze_liquidity('AAPL')
        
        # Verify neutral score is returned
        assert result['anomaly_score'] == 0.5
        assert result['status'] == 'Insufficient data for microstructure analysis'
        assert result['confidence'] == 'low'
    
    def test_analyze_liquidity_with_empty_dataframe(self):
        """Test analyze_liquidity handles empty DataFrame gracefully."""
        mock_df = pd.DataFrame()
        
        with patch('src.market.get_intraday_data', return_value=mock_df):
            result = analyze_liquidity('AAPL')
        
        # Verify neutral score is returned
        assert result['anomaly_score'] == 0.5
        assert result['status'] == 'Insufficient data for microstructure analysis'
        assert result['confidence'] == 'low'
    
    def test_analyze_liquidity_classification_critical_void(self):
        """Test classification with score > 0.85 returns 'Critical liquidity void'."""
        # Create mock DataFrame
        dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='1min')
        mock_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Mock the CNN model to return a high score
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.item.return_value = 0.92
        mock_model.return_value = mock_output
        
        with patch('src.market.get_intraday_data', return_value=mock_df):
            with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
                result = analyze_liquidity('AAPL')
        
        assert result['anomaly_score'] == 0.92
        assert result['status'] == 'Critical liquidity void detected'
        assert result['confidence'] == 'high'
    
    def test_analyze_liquidity_classification_high_volatility(self):
        """Test classification with 0.70 < score <= 0.85 returns 'High microstructure volatility'."""
        dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='1min')
        mock_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.item.return_value = 0.75
        mock_model.return_value = mock_output
        
        with patch('src.market.get_intraday_data', return_value=mock_df):
            with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
                result = analyze_liquidity('AAPL')
        
        assert result['anomaly_score'] == 0.75
        assert result['status'] == 'High microstructure volatility'
        assert result['confidence'] == 'high'
    
    def test_analyze_liquidity_classification_normal_conditions(self):
        """Test classification with 0.20 <= score <= 0.70 returns 'Normal market conditions'."""
        dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='1min')
        mock_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.item.return_value = 0.5
        mock_model.return_value = mock_output
        
        with patch('src.market.get_intraday_data', return_value=mock_df):
            with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
                result = analyze_liquidity('AAPL')
        
        assert result['anomaly_score'] == 0.5
        assert result['status'] == 'Normal market conditions'
        assert result['confidence'] == 'high'
    
    def test_analyze_liquidity_classification_accumulation_zone(self):
        """Test classification with score < 0.20 returns 'Strong accumulation zone'."""
        dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='1min')
        mock_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.item.return_value = 0.15
        mock_model.return_value = mock_output
        
        with patch('src.market.get_intraday_data', return_value=mock_df):
            with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
                result = analyze_liquidity('AAPL')
        
        assert result['anomaly_score'] == 0.15
        assert result['status'] == 'Strong accumulation zone detected'
        assert result['confidence'] == 'high'
    
    def test_analyze_liquidity_handles_heatmap_failure(self):
        """Test analyze_liquidity handles heatmap generation failure gracefully."""
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
                result = analyze_liquidity('AAPL')
        
        # Verify neutral score is returned
        assert result['anomaly_score'] == 0.5
        assert result['status'] == 'Insufficient data for microstructure analysis'
        assert result['confidence'] == 'low'
    
    def test_analyze_liquidity_handles_exception(self):
        """Test analyze_liquidity handles unexpected exceptions gracefully."""
        with patch('src.market.get_intraday_data', side_effect=Exception('API error')):
            result = analyze_liquidity('AAPL')
        
        # Verify neutral score is returned
        assert result['anomaly_score'] == 0.5
        assert result['status'] == 'Insufficient data for microstructure analysis'
        assert result['confidence'] == 'low'

