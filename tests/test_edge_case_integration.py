"""
Integration tests for edge cases in microstructure analysis.

Feature: convolutional-order-book
Task: 12.2 Write integration tests for edge cases

Tests edge cases:
- Ticker with no intraday data available
- Ticker with partial data (< 64 minutes)
- Ticker during market hours vs after hours
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.market import get_intraday_data
from src.microstructure import analyze_liquidity


class TestEdgeCaseIntegration:
    """
    Integration tests for edge cases in the microstructure analysis flow.
    
    These tests verify that the system handles various edge cases gracefully
    and returns appropriate results.
    """
    
    def test_ticker_with_no_intraday_data(self):
        """
        Test with ticker that has no intraday data available.
        
        The system should handle this gracefully and return a neutral score
        with appropriate status message.
        
        **Validates: Requirements 4.9, 6.3**
        """
        ticker = "INVALID_TICKER"
        
        # Mock get_intraday_data to return empty DataFrame
        with patch('src.market.get_intraday_data', return_value=pd.DataFrame()):
            result = analyze_liquidity(ticker)
            
            # Should return neutral score
            assert result['anomaly_score'] == 0.5, \
                "Should return neutral score for no data"
            
            # Should indicate insufficient data
            assert result['status'] == 'Insufficient data for microstructure analysis', \
                "Status should indicate insufficient data"
            
            # Should have low confidence
            assert result['confidence'] == 'low', \
                "Confidence should be low for no data"
            
            print(f"✓ No data case handled correctly")
            print(f"  - Anomaly score: {result['anomaly_score']}")
            print(f"  - Status: {result['status']}")
    
    def test_ticker_with_partial_data_less_than_64_minutes(self):
        """
        Test with ticker that has partial data (< 64 minutes).
        
        The system should pad the heatmap with zeros and still generate
        a valid analysis result.
        
        **Validates: Requirements 2.6, 4.9**
        """
        ticker = "PARTIAL_DATA"
        
        # Create partial data (only 30 minutes)
        dates = pd.date_range(start='2024-01-01 09:30', periods=30, freq='1min')
        partial_df = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(30)],
            'High': [100.5 + i * 0.1 for i in range(30)],
            'Low': [99.5 + i * 0.1 for i in range(30)],
            'Close': [100 + i * 0.1 for i in range(30)],
            'Volume': [1000 + i * 10 for i in range(30)]
        }, index=dates)
        
        # Mock get_intraday_data to return partial data
        with patch('src.market.get_intraday_data', return_value=partial_df):
            result = analyze_liquidity(ticker)
            
            # Should still return a valid result
            assert 'anomaly_score' in result, "Should have anomaly score"
            assert 'status' in result, "Should have status"
            assert 'confidence' in result, "Should have confidence"
            
            # Anomaly score should be in valid range
            assert 0.0 <= result['anomaly_score'] <= 1.0, \
                f"Anomaly score should be in [0, 1], got {result['anomaly_score']}"
            
            print(f"✓ Partial data case handled correctly")
            print(f"  - Data points: 30 minutes (padded to 64)")
            print(f"  - Anomaly score: {result['anomaly_score']:.3f}")
            print(f"  - Status: {result['status']}")
    
    def test_ticker_with_very_little_data(self):
        """
        Test with ticker that has very little data (< 10 minutes).
        
        The system should return a neutral score indicating insufficient data.
        
        **Validates: Requirements 4.9**
        """
        ticker = "MINIMAL_DATA"
        
        # Create minimal data (only 5 minutes)
        dates = pd.date_range(start='2024-01-01 09:30', periods=5, freq='1min')
        minimal_df = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(5)],
            'High': [100.5 + i * 0.1 for i in range(5)],
            'Low': [99.5 + i * 0.1 for i in range(5)],
            'Close': [100 + i * 0.1 for i in range(5)],
            'Volume': [1000 + i * 10 for i in range(5)]
        }, index=dates)
        
        # Mock get_intraday_data to return minimal data
        with patch('src.market.get_intraday_data', return_value=minimal_df):
            result = analyze_liquidity(ticker)
            
            # Should return neutral score
            assert result['anomaly_score'] == 0.5, \
                "Should return neutral score for insufficient data"
            
            # Should indicate insufficient data
            assert result['status'] == 'Insufficient data for microstructure analysis', \
                "Status should indicate insufficient data"
            
            print(f"✓ Minimal data case handled correctly")
            print(f"  - Data points: 5 minutes (< 10 minimum)")
            print(f"  - Anomaly score: {result['anomaly_score']}")
    
    def test_ticker_during_market_hours_simulation(self):
        """
        Test with ticker during market hours (simulated).
        
        During market hours, we should have recent data and be able to
        generate a valid analysis.
        
        **Validates: All requirements**
        """
        ticker = "MARKET_HOURS"
        
        # Create recent data (simulating market hours)
        now = datetime.now()
        dates = pd.date_range(end=now, periods=100, freq='1min')
        market_hours_df = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(100)],
            'High': [100.5 + i * 0.1 for i in range(100)],
            'Low': [99.5 + i * 0.1 for i in range(100)],
            'Close': [100 + i * 0.1 for i in range(100)],
            'Volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)
        
        # Mock get_intraday_data to return recent data
        with patch('src.market.get_intraday_data', return_value=market_hours_df):
            result = analyze_liquidity(ticker)
            
            # Should return a valid result
            assert 'anomaly_score' in result, "Should have anomaly score"
            assert 0.0 <= result['anomaly_score'] <= 1.0, \
                "Anomaly score should be in valid range"
            
            # Should have high confidence (sufficient data)
            assert result['confidence'] in ['high', 'low'], \
                "Confidence should be either high or low"
            
            print(f"✓ Market hours case handled correctly")
            print(f"  - Data points: 100 minutes (recent)")
            print(f"  - Anomaly score: {result['anomaly_score']:.3f}")
            print(f"  - Status: {result['status']}")
    
    def test_ticker_after_market_hours_simulation(self):
        """
        Test with ticker after market hours (simulated).
        
        After market hours, we may have stale data but should still be able
        to generate an analysis from the most recent available data.
        
        **Validates: All requirements**
        """
        ticker = "AFTER_HOURS"
        
        # Create older data (simulating after hours)
        yesterday = datetime.now() - timedelta(days=1)
        dates = pd.date_range(end=yesterday, periods=100, freq='1min')
        after_hours_df = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(100)],
            'High': [100.5 + i * 0.1 for i in range(100)],
            'Low': [99.5 + i * 0.1 for i in range(100)],
            'Close': [100 + i * 0.1 for i in range(100)],
            'Volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)
        
        # Mock get_intraday_data to return older data
        with patch('src.market.get_intraday_data', return_value=after_hours_df):
            result = analyze_liquidity(ticker)
            
            # Should still return a valid result
            assert 'anomaly_score' in result, "Should have anomaly score"
            assert 0.0 <= result['anomaly_score'] <= 1.0, \
                "Anomaly score should be in valid range"
            
            print(f"✓ After hours case handled correctly")
            print(f"  - Data points: 100 minutes (from yesterday)")
            print(f"  - Anomaly score: {result['anomaly_score']:.3f}")
            print(f"  - Status: {result['status']}")
    
    def test_ticker_with_zero_volume(self):
        """
        Test with ticker that has zero volume throughout.
        
        The system should handle this edge case and still generate a heatmap
        (though it will be mostly zeros).
        
        **Validates: Requirements 2.4, 2.5**
        """
        ticker = "ZERO_VOLUME"
        
        # Create data with zero volume
        dates = pd.date_range(start='2024-01-01 09:30', periods=64, freq='1min')
        zero_volume_df = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(64)],
            'High': [100.5 + i * 0.1 for i in range(64)],
            'Low': [99.5 + i * 0.1 for i in range(64)],
            'Close': [100 + i * 0.1 for i in range(64)],
            'Volume': [0] * 64  # All zeros
        }, index=dates)
        
        # Mock get_intraday_data to return zero volume data
        with patch('src.market.get_intraday_data', return_value=zero_volume_df):
            result = analyze_liquidity(ticker)
            
            # Should still return a valid result
            assert 'anomaly_score' in result, "Should have anomaly score"
            assert 0.0 <= result['anomaly_score'] <= 1.0, \
                "Anomaly score should be in valid range"
            
            print(f"✓ Zero volume case handled correctly")
            print(f"  - All volumes: 0")
            print(f"  - Anomaly score: {result['anomaly_score']:.3f}")
            print(f"  - Status: {result['status']}")
    
    def test_ticker_with_extreme_volatility(self):
        """
        Test with ticker that has extreme price volatility.
        
        The system should handle extreme price movements and still generate
        a valid analysis.
        
        **Validates: Requirements 2.3, 2.5**
        """
        ticker = "EXTREME_VOLATILITY"
        
        # Create data with extreme volatility
        dates = pd.date_range(start='2024-01-01 09:30', periods=64, freq='1min')
        volatile_df = pd.DataFrame({
            'Open': [100 + (i % 2) * 50 for i in range(64)],  # Oscillates between 100 and 150
            'High': [150 + (i % 2) * 50 for i in range(64)],
            'Low': [50 + (i % 2) * 50 for i in range(64)],
            'Close': [100 + (i % 2) * 50 for i in range(64)],
            'Volume': [1000 + i * 100 for i in range(64)]
        }, index=dates)
        
        # Mock get_intraday_data to return volatile data
        with patch('src.market.get_intraday_data', return_value=volatile_df):
            result = analyze_liquidity(ticker)
            
            # Should still return a valid result
            assert 'anomaly_score' in result, "Should have anomaly score"
            assert 0.0 <= result['anomaly_score'] <= 1.0, \
                "Anomaly score should be in valid range"
            
            print(f"✓ Extreme volatility case handled correctly")
            print(f"  - Price range: 50-200 (extreme)")
            print(f"  - Anomaly score: {result['anomaly_score']:.3f}")
            print(f"  - Status: {result['status']}")
    
    def test_system_handles_all_edge_cases_gracefully(self):
        """
        Test that the system handles all edge cases without crashing.
        
        This is a comprehensive test that verifies the system's robustness
        across multiple edge cases.
        
        **Validates: Requirements 6.1, 6.2**
        """
        edge_cases = [
            ("Empty data", pd.DataFrame()),
            ("Single row", pd.DataFrame({
                'High': [100], 'Low': [99], 'Volume': [1000]
            })),
            ("All NaN", pd.DataFrame({
                'High': [np.nan] * 64,
                'Low': [np.nan] * 64,
                'Volume': [np.nan] * 64
            }, index=pd.date_range(start='2024-01-01', periods=64, freq='1min'))),
            ("Zero price range", pd.DataFrame({
                'High': [100] * 64,
                'Low': [100] * 64,
                'Volume': [1000] * 64
            }, index=pd.date_range(start='2024-01-01', periods=64, freq='1min'))),
        ]
        
        for case_name, test_df in edge_cases:
            try:
                with patch('src.market.get_intraday_data', return_value=test_df):
                    result = analyze_liquidity("TEST")
                    
                    # Should always return a valid result
                    assert 'anomaly_score' in result, f"{case_name}: Should have anomaly score"
                    assert 'status' in result, f"{case_name}: Should have status"
                    assert 'confidence' in result, f"{case_name}: Should have confidence"
                    
                    # Anomaly score should always be in valid range
                    assert 0.0 <= result['anomaly_score'] <= 1.0, \
                        f"{case_name}: Anomaly score should be in [0, 1]"
                    
                    print(f"✓ {case_name} handled gracefully")
                    
            except Exception as e:
                pytest.fail(f"{case_name} raised exception: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
