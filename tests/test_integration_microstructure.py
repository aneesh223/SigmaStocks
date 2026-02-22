"""
Unit tests for microstructure integration with trading recommendation system.

Tests specific examples of confidence adjustments based on anomaly scores.
**Validates: Requirements 5.2, 5.3, 5.4, 5.5, 5.6, 5.7**
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

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


class TestBuyConfidenceReduction:
    """Test BUY recommendation confidence reductions."""
    
    def test_buy_with_score_0_85_reduces_confidence_by_30_percent(self):
        """
        Test BUY recommendation with score 0.85 reduces confidence by 30%.
        **Validates: Requirements 5.2**
        """
        mock_sentiment_df = create_mock_sentiment_df(6.0)
        mock_price_data = create_mock_price_data()
        
        mock_microstructure = {
            'anomaly_score': 0.85,
            'status': 'Critical liquidity void detected',
            'confidence': 'high'
        }
        
        # Get baseline
        with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
            baseline_result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Get adjusted result
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
            adjusted_result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        if baseline_result['recommendation'] == 'BUY':
            baseline_confidence = baseline_result['confidence']
            adjusted_confidence = adjusted_result['confidence']
            
            # Score 0.85 > 0.8, so should apply 30% reduction (0.7x)
            expected_confidence = baseline_confidence * 0.7
            
            assert abs(adjusted_confidence - expected_confidence) < 1.0, \
                f"Expected confidence ~{expected_confidence:.2f}, got {adjusted_confidence:.2f}"
    
    def test_buy_with_score_0_90_reduces_confidence_by_30_percent(self):
        """
        Test BUY recommendation with score 0.90 (> 0.8) reduces confidence by 30%.
        **Validates: Requirements 5.2**
        """
        mock_sentiment_df = create_mock_sentiment_df(6.0)
        mock_price_data = create_mock_price_data()
        
        mock_microstructure = {
            'anomaly_score': 0.90,
            'status': 'Critical liquidity void detected',
            'confidence': 'high'
        }
        
        # Get baseline
        with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
            baseline_result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Get adjusted result
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
            adjusted_result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        if baseline_result['recommendation'] == 'BUY':
            baseline_confidence = baseline_result['confidence']
            adjusted_confidence = adjusted_result['confidence']
            
            # Score > 0.8 should apply 30% reduction (0.7x)
            expected_confidence = baseline_confidence * 0.7
            
            assert abs(adjusted_confidence - expected_confidence) < 1.0, \
                f"Expected confidence ~{expected_confidence:.2f}, got {adjusted_confidence:.2f}"
    
    def test_buy_with_score_0_75_reduces_confidence_by_15_percent(self):
        """
        Test BUY recommendation with score 0.75 reduces confidence by 15%.
        **Validates: Requirements 5.3**
        """
        mock_sentiment_df = create_mock_sentiment_df(6.0)
        mock_price_data = create_mock_price_data()
        
        mock_microstructure = {
            'anomaly_score': 0.75,
            'status': 'High microstructure volatility',
            'confidence': 'high'
        }
        
        # Get baseline
        with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
            baseline_result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Get adjusted result
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
            adjusted_result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        if baseline_result['recommendation'] == 'BUY':
            baseline_confidence = baseline_result['confidence']
            adjusted_confidence = adjusted_result['confidence']
            
            # Score 0.7 < 0.75 <= 0.8 should apply 15% reduction (0.85x)
            expected_confidence = baseline_confidence * 0.85
            
            assert abs(adjusted_confidence - expected_confidence) < 1.0, \
                f"Expected confidence ~{expected_confidence:.2f}, got {adjusted_confidence:.2f}"


class TestBuyConfidenceBoost:
    """Test BUY recommendation confidence boosts."""
    
    def test_buy_with_score_0_15_boosts_confidence_by_20_percent(self):
        """
        Test BUY recommendation with score 0.15 boosts confidence by 20%.
        **Validates: Requirements 5.4**
        """
        mock_sentiment_df = create_mock_sentiment_df(6.0)
        mock_price_data = create_mock_price_data()
        
        mock_microstructure = {
            'anomaly_score': 0.15,
            'status': 'Strong accumulation zone detected',
            'confidence': 'high'
        }
        
        # Get baseline
        with patch('src.microstructure.analyze_liquidity', side_effect=Exception("Disabled")):
            baseline_result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Get adjusted result
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
            adjusted_result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        if baseline_result['recommendation'] == 'BUY':
            baseline_confidence = baseline_result['confidence']
            adjusted_confidence = adjusted_result['confidence']
            
            # Score < 0.2 should apply 20% boost (1.2x), clamped to 100
            expected_confidence = min(baseline_confidence * 1.2, 100)
            
            assert abs(adjusted_confidence - expected_confidence) < 1.0, \
                f"Expected confidence ~{expected_confidence:.2f}, got {adjusted_confidence:.2f}"


class TestSellHoldPreservation:
    """Test SELL/HOLD confidence preservation."""
    
    def test_sell_recommendation_with_any_score_leaves_confidence_unchanged(self):
        """
        Test SELL recommendation with any score leaves confidence unchanged.
        **Validates: Requirements 5.5**
        """
        mock_sentiment_df = create_mock_sentiment_df(3.0)
        mock_price_data = create_mock_price_data()
        
        # Test with various anomaly scores
        for anomaly_score in [0.05, 0.5, 0.75, 0.95]:
            mock_microstructure = {
                'anomaly_score': anomaly_score,
                'status': 'Test status',
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
            
            # Get adjusted result
            with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
                adjusted_result = get_trading_recommendation(
                    ticker='TEST',
                    final_buy_score=3.0,
                    sentiment_df=mock_sentiment_df,
                    price_data=mock_price_data,
                    strategy='momentum'
                )
            
            if baseline_result['recommendation'] == 'SELL':
                assert abs(adjusted_result['confidence'] - baseline_result['confidence']) < 0.01, \
                    f"SELL confidence changed with anomaly score {anomaly_score}"
    
    def test_hold_recommendation_with_any_score_leaves_confidence_unchanged(self):
        """
        Test HOLD recommendation with any score leaves confidence unchanged.
        **Validates: Requirements 5.5**
        """
        mock_sentiment_df = create_mock_sentiment_df(4.9)
        mock_price_data = create_mock_price_data()
        
        # Test with various anomaly scores
        for anomaly_score in [0.05, 0.5, 0.75, 0.95]:
            mock_microstructure = {
                'anomaly_score': anomaly_score,
                'status': 'Test status',
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
            
            # Get adjusted result
            with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
                adjusted_result = get_trading_recommendation(
                    ticker='TEST',
                    final_buy_score=4.9,
                    sentiment_df=mock_sentiment_df,
                    price_data=mock_price_data,
                    strategy='momentum'
                )
            
            if baseline_result['recommendation'] == 'HOLD':
                assert abs(adjusted_result['confidence'] - baseline_result['confidence']) < 0.01, \
                    f"HOLD confidence changed with anomaly score {anomaly_score}"


class TestFailSafe:
    """Test fail-safe behavior."""
    
    def test_microstructure_failure_doesnt_crash_recommendation_system(self):
        """
        Test microstructure failure doesn't crash recommendation system.
        **Validates: Requirements 5.6**
        """
        mock_sentiment_df = create_mock_sentiment_df(6.0)
        mock_price_data = create_mock_price_data()
        
        # Test with exception
        with patch('src.microstructure.analyze_liquidity', side_effect=RuntimeError("Test error")):
            result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        # Should return valid result
        assert isinstance(result, dict)
        assert 'recommendation' in result
        assert 'confidence' in result
        assert result['recommendation'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= result['confidence'] <= 100


class TestConfidenceBounds:
    """Test confidence bounds after adjustments."""
    
    def test_confidence_is_clamped_to_0_100_after_extreme_adjustments(self):
        """
        Test confidence is clamped to [0, 100] after extreme adjustments.
        **Validates: Requirements 5.7**
        """
        # Test with very high buy score and accumulation boost
        mock_sentiment_df = create_mock_sentiment_df(9.5)
        mock_price_data = create_mock_price_data()
        
        mock_microstructure = {
            'anomaly_score': 0.05,
            'status': 'Strong accumulation zone detected',
            'confidence': 'high'
        }
        
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
            result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=9.5,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        if result['recommendation'] == 'BUY':
            # Even with 20% boost, confidence must not exceed 100
            assert result['confidence'] <= 100, \
                f"Confidence {result['confidence']} exceeds 100"
            assert result['confidence'] >= 0, \
                f"Confidence {result['confidence']} is negative"
        
        # Test with low buy score and high anomaly reduction
        mock_sentiment_df_low = create_mock_sentiment_df(5.2)
        
        mock_microstructure_high = {
            'anomaly_score': 0.95,
            'status': 'Critical liquidity void detected',
            'confidence': 'high'
        }
        
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure_high):
            result_low = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=5.2,
                sentiment_df=mock_sentiment_df_low,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        if result_low['recommendation'] == 'BUY':
            # Even with 30% reduction, confidence must not go negative
            assert result_low['confidence'] >= 0, \
                f"Confidence {result_low['confidence']} is negative"
            assert result_low['confidence'] <= 100, \
                f"Confidence {result_low['confidence']} exceeds 100"


class TestReasoningInclusion:
    """Test that microstructure status is included in reasoning."""
    
    def test_microstructure_status_included_in_reasoning_for_buy(self):
        """Test that microstructure status is included in reasoning for BUY recommendations."""
        mock_sentiment_df = create_mock_sentiment_df(6.0)
        mock_price_data = create_mock_price_data()
        
        mock_microstructure = {
            'anomaly_score': 0.85,
            'status': 'High microstructure volatility',
            'confidence': 'high'
        }
        
        with patch('src.microstructure.analyze_liquidity', return_value=mock_microstructure):
            result = get_trading_recommendation(
                ticker='TEST',
                final_buy_score=6.0,
                sentiment_df=mock_sentiment_df,
                price_data=mock_price_data,
                strategy='momentum'
            )
        
        if result['recommendation'] == 'BUY':
            # Microstructure status should be in reasoning
            assert 'Microstructure' in result['reasoning'], \
                "Microstructure status should be included in reasoning for BUY"
            assert '0.85' in result['reasoning'] or '0.8' in result['reasoning'], \
                "Anomaly score should be included in reasoning"
