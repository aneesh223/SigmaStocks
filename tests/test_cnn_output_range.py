"""
Property-based tests for CNN output range invariant.

Feature: convolutional-order-book
Task: 3.4 Write property test for CNN output range
"""

import pytest
import torch
from hypothesis import given, strategies as st

from src.microstructure import OrderBookCNN


@given(
    batch_size=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1)
)
def test_cnn_output_range_invariant(batch_size, seed):
    """
    Feature: convolutional-order-book, Property 7: CNN Output Range Invariant
    **Validates: Requirements 3.4**
    
    Test that for any valid input heatmap tensor of shape (batch_size, 1, 64, 64),
    the OrderBookCNN output is a tensor with all values in the range [0.0, 1.0]
    due to the Sigmoid activation function.
    
    This property must hold for:
    - Any batch size from 1 to 32
    - Any random tensor values (including extreme values, NaN handling by model)
    - Both positive and negative input values
    - Zero tensors and uniform tensors
    """
    # Set seed for reproducibility within this test iteration
    torch.manual_seed(seed)
    
    # Create model in evaluation mode
    model = OrderBookCNN()
    model.eval()
    
    # Generate random input tensor of shape (batch_size, 1, 64, 64)
    # Use randn to get values from standard normal distribution
    # This tests the model with a wide range of input values
    input_tensor = torch.randn(batch_size, 1, 64, 64)
    
    # Run inference without gradient computation
    with torch.no_grad():
        output = model(input_tensor)
    
    # Verify output shape is correct
    assert output.shape == (batch_size, 1), \
        f"Expected output shape ({batch_size}, 1), but got {output.shape}"
    
    # Verify all output values are in [0.0, 1.0] range
    assert torch.all(output >= 0.0), \
        f"All output values must be >= 0.0, but found min value: {output.min().item()}"
    
    assert torch.all(output <= 1.0), \
        f"All output values must be <= 1.0, but found max value: {output.max().item()}"
    
    # Verify output contains no NaN or Inf values
    assert not torch.any(torch.isnan(output)), \
        "Output should not contain NaN values"
    
    assert not torch.any(torch.isinf(output)), \
        "Output should not contain Inf values"


@given(
    batch_size=st.integers(min_value=1, max_value=16),
    value=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
def test_cnn_output_range_with_uniform_input(batch_size, value):
    """
    Feature: convolutional-order-book, Property 7: CNN Output Range Invariant
    **Validates: Requirements 3.4**
    
    Test that CNN output is in [0.0, 1.0] even with uniform input tensors.
    This tests edge cases where all input values are identical.
    """
    # Create model in evaluation mode
    model = OrderBookCNN()
    model.eval()
    
    # Create uniform input tensor (all values are the same)
    input_tensor = torch.full((batch_size, 1, 64, 64), value)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Verify output is in valid range
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), \
        f"Output must be in [0.0, 1.0], but got values in range [{output.min().item()}, {output.max().item()}]"


@given(
    batch_size=st.integers(min_value=1, max_value=16)
)
def test_cnn_output_range_with_extreme_values(batch_size):
    """
    Feature: convolutional-order-book, Property 7: CNN Output Range Invariant
    **Validates: Requirements 3.4**
    
    Test that CNN output is in [0.0, 1.0] even with extreme input values.
    This tests the model's robustness to outliers and extreme inputs.
    """
    model = OrderBookCNN()
    model.eval()
    
    # Test with very large positive values
    input_large = torch.full((batch_size, 1, 64, 64), 1e6)
    with torch.no_grad():
        output_large = model(input_large)
    assert torch.all(output_large >= 0.0) and torch.all(output_large <= 1.0), \
        "Output must be in [0.0, 1.0] for large positive inputs"
    
    # Test with very large negative values
    input_small = torch.full((batch_size, 1, 64, 64), -1e6)
    with torch.no_grad():
        output_small = model(input_small)
    assert torch.all(output_small >= 0.0) and torch.all(output_small <= 1.0), \
        "Output must be in [0.0, 1.0] for large negative inputs"
    
    # Test with zero tensor
    input_zero = torch.zeros((batch_size, 1, 64, 64))
    with torch.no_grad():
        output_zero = model(input_zero)
    assert torch.all(output_zero >= 0.0) and torch.all(output_zero <= 1.0), \
        "Output must be in [0.0, 1.0] for zero input"


def test_cnn_output_range_single_sample():
    """
    Unit test: Verify output range for a single sample.
    **Validates: Requirements 3.4**
    """
    model = OrderBookCNN()
    model.eval()
    
    # Single sample
    input_tensor = torch.randn(1, 1, 64, 64)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Verify shape
    assert output.shape == (1, 1), f"Expected shape (1, 1), got {output.shape}"
    
    # Verify range
    output_value = output.item()
    assert 0.0 <= output_value <= 1.0, \
        f"Output value {output_value} is not in range [0.0, 1.0]"


def test_cnn_output_range_large_batch():
    """
    Unit test: Verify output range for a large batch.
    **Validates: Requirements 3.4**
    """
    model = OrderBookCNN()
    model.eval()
    
    # Large batch
    batch_size = 64
    input_tensor = torch.randn(batch_size, 1, 64, 64)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Verify shape
    assert output.shape == (batch_size, 1), \
        f"Expected shape ({batch_size}, 1), got {output.shape}"
    
    # Verify all values are in range
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), \
        f"All outputs must be in [0.0, 1.0], got range [{output.min().item()}, {output.max().item()}]"
    
    # Verify no NaN or Inf
    assert not torch.any(torch.isnan(output)), "Output contains NaN values"
    assert not torch.any(torch.isinf(output)), "Output contains Inf values"
