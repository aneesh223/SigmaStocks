"""
Property-based test for classification threshold correctness.

**Property 8: Classification Threshold Correctness**
**Validates: Requirements 4.5, 4.6, 4.7, 4.8**

For any anomaly score value, the status message MUST correctly reflect the score's 
threshold range:
- score > 0.85 → "Critical liquidity void detected"
- 0.70 < score <= 0.85 → "High microstructure volatility"
- 0.20 <= score <= 0.70 → "Normal market conditions"
- score < 0.20 → "Strong accumulation zone detected"
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta

from src.microstructure import analyze_liquidity


def create_mock_dataframe(num_rows=100):
    """Helper function to create a mock OHLCV DataFrame."""
    dates = pd.date_range(start='2024-01-01 09:30', periods=num_rows, freq='1min')
    return pd.DataFrame({
        'Open': np.random.uniform(100, 110, num_rows),
        'High': np.random.uniform(110, 120, num_rows),
        'Low': np.random.uniform(90, 100, num_rows),
        'Close': np.random.uniform(100, 110, num_rows),
        'Volume': np.random.randint(1000000, 5000000, num_rows)
    }, index=dates)


@given(score=st.floats(min_value=0.851, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_classification_critical_liquidity_void(score):
    """
    **Property 8: Classification Threshold Correctness (Critical Void)**
    **Validates: Requirements 4.5**
    
    For any anomaly score > 0.85, the status message MUST be 
    "Critical liquidity void detected".
    """
    mock_df = create_mock_dataframe()
    
    # Mock the CNN model to return the specific score
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.item.return_value = score
    mock_model.return_value = mock_output
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    # Verify the score is correctly returned
    assert result['anomaly_score'] == score
    
    # CRITICAL: Verify the status message matches the threshold
    assert result['status'] == 'Critical liquidity void detected', \
        f"Score {score} > 0.85 should return 'Critical liquidity void detected', got '{result['status']}'"
    
    # Verify confidence is high
    assert result['confidence'] == 'high'


@given(score=st.floats(min_value=0.701, max_value=0.85, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_classification_high_microstructure_volatility(score):
    """
    **Property 8: Classification Threshold Correctness (High Volatility)**
    **Validates: Requirements 4.6**
    
    For any anomaly score in (0.70, 0.85], the status message MUST be 
    "High microstructure volatility".
    """
    mock_df = create_mock_dataframe()
    
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.item.return_value = score
    mock_model.return_value = mock_output
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    assert result['anomaly_score'] == score
    
    # CRITICAL: Verify the status message matches the threshold
    assert result['status'] == 'High microstructure volatility', \
        f"Score {score} in (0.70, 0.85] should return 'High microstructure volatility', got '{result['status']}'"
    
    assert result['confidence'] == 'high'


@given(score=st.floats(min_value=0.20, max_value=0.70, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_classification_normal_market_conditions(score):
    """
    **Property 8: Classification Threshold Correctness (Normal Conditions)**
    **Validates: Requirements 4.7**
    
    For any anomaly score in [0.20, 0.70], the status message MUST be 
    "Normal market conditions".
    """
    mock_df = create_mock_dataframe()
    
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.item.return_value = score
    mock_model.return_value = mock_output
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    assert result['anomaly_score'] == score
    
    # CRITICAL: Verify the status message matches the threshold
    assert result['status'] == 'Normal market conditions', \
        f"Score {score} in [0.20, 0.70] should return 'Normal market conditions', got '{result['status']}'"
    
    assert result['confidence'] == 'high'


@given(score=st.floats(min_value=0.0, max_value=0.199, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_classification_strong_accumulation_zone(score):
    """
    **Property 8: Classification Threshold Correctness (Accumulation Zone)**
    **Validates: Requirements 4.8**
    
    For any anomaly score < 0.20, the status message MUST be 
    "Strong accumulation zone detected".
    """
    mock_df = create_mock_dataframe()
    
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.item.return_value = score
    mock_model.return_value = mock_output
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    assert result['anomaly_score'] == score
    
    # CRITICAL: Verify the status message matches the threshold
    assert result['status'] == 'Strong accumulation zone detected', \
        f"Score {score} < 0.20 should return 'Strong accumulation zone detected', got '{result['status']}'"
    
    assert result['confidence'] == 'high'


# Boundary tests for exact threshold values
def test_classification_boundary_0_85():
    """Test exact boundary at score = 0.85 (should be High volatility, not Critical void)."""
    mock_df = create_mock_dataframe()
    
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.item.return_value = 0.85
    mock_model.return_value = mock_output
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    assert result['anomaly_score'] == 0.85
    # At exactly 0.85, should be "High microstructure volatility" (not > 0.85)
    assert result['status'] == 'High microstructure volatility', \
        "Score exactly 0.85 should return 'High microstructure volatility'"


def test_classification_boundary_0_70():
    """Test exact boundary at score = 0.70 (should be Normal conditions, not High volatility)."""
    mock_df = create_mock_dataframe()
    
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.item.return_value = 0.70
    mock_model.return_value = mock_output
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    assert result['anomaly_score'] == 0.70
    # At exactly 0.70, should be "Normal market conditions" (not > 0.70)
    assert result['status'] == 'Normal market conditions', \
        "Score exactly 0.70 should return 'Normal market conditions'"


def test_classification_boundary_0_20():
    """Test exact boundary at score = 0.20 (should be Normal conditions, not Accumulation)."""
    mock_df = create_mock_dataframe()
    
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.item.return_value = 0.20
    mock_model.return_value = mock_output
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    assert result['anomaly_score'] == 0.20
    # At exactly 0.20, should be "Normal market conditions" (>= 0.20)
    assert result['status'] == 'Normal market conditions', \
        "Score exactly 0.20 should return 'Normal market conditions'"


def test_classification_extreme_values():
    """Test extreme boundary values (0.0 and 1.0)."""
    mock_df = create_mock_dataframe()
    
    # Test score = 0.0
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.item.return_value = 0.0
    mock_model.return_value = mock_output
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    assert result['anomaly_score'] == 0.0
    assert result['status'] == 'Strong accumulation zone detected', \
        "Score 0.0 should return 'Strong accumulation zone detected'"
    
    # Test score = 1.0
    mock_output.item.return_value = 1.0
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    assert result['anomaly_score'] == 1.0
    assert result['status'] == 'Critical liquidity void detected', \
        "Score 1.0 should return 'Critical liquidity void detected'"


@given(score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, deadline=None)
def test_classification_completeness(score):
    """
    **Property 8: Classification Threshold Correctness (Completeness)**
    **Validates: Requirements 4.5, 4.6, 4.7, 4.8**
    
    For ANY score in [0.0, 1.0], there MUST be exactly one valid status message.
    This ensures complete coverage of the score range with no gaps.
    """
    mock_df = create_mock_dataframe()
    
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.item.return_value = score
    mock_model.return_value = mock_output
    
    with patch('src.market.get_intraday_data', return_value=mock_df):
        with patch('src.microstructure._get_model', return_value=(mock_model, 'cpu')):
            result = analyze_liquidity('TEST')
    
    # Verify a status is returned
    assert 'status' in result
    status = result['status']
    
    # Verify it's one of the valid status messages
    valid_statuses = [
        'Critical liquidity void detected',
        'High microstructure volatility',
        'Normal market conditions',
        'Strong accumulation zone detected'
    ]
    
    assert status in valid_statuses, \
        f"Score {score} returned invalid status '{status}'"
    
    # Verify the status matches the expected threshold
    if score > 0.85:
        expected = 'Critical liquidity void detected'
    elif score > 0.70:
        expected = 'High microstructure volatility'
    elif score >= 0.20:
        expected = 'Normal market conditions'
    else:
        expected = 'Strong accumulation zone detected'
    
    assert status == expected, \
        f"Score {score} returned '{status}' but expected '{expected}'"
