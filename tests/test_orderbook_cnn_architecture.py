"""
Unit tests for OrderBookCNN architecture.

Feature: convolutional-order-book
Task: 3.3 Write unit tests for OrderBookCNN architecture
"""

import pytest
import torch
import torch.nn as nn

from src.microstructure import OrderBookCNN


def test_model_has_correct_number_of_layers():
    """
    Test that OrderBookCNN has the correct layer structure.
    **Validates: Requirements 3.1, 3.2**
    """
    model = OrderBookCNN()
    
    # Count the main layers
    # Expected: 2 Conv2d, 2 BatchNorm2d, 2 ReLU, 2 MaxPool2d, 2 Dropout, 1 Flatten, 1 Linear, 1 Sigmoid
    assert hasattr(model, 'conv1'), "Model should have conv1 layer"
    assert hasattr(model, 'bn1'), "Model should have bn1 layer"
    assert hasattr(model, 'relu1'), "Model should have relu1 layer"
    assert hasattr(model, 'pool1'), "Model should have pool1 layer"
    assert hasattr(model, 'dropout1'), "Model should have dropout1 layer"
    
    assert hasattr(model, 'conv2'), "Model should have conv2 layer"
    assert hasattr(model, 'bn2'), "Model should have bn2 layer"
    assert hasattr(model, 'relu2'), "Model should have relu2 layer"
    assert hasattr(model, 'pool2'), "Model should have pool2 layer"
    assert hasattr(model, 'dropout2'), "Model should have dropout2 layer"
    
    assert hasattr(model, 'flatten'), "Model should have flatten layer"
    assert hasattr(model, 'fc'), "Model should have fc (fully connected) layer"
    assert hasattr(model, 'sigmoid'), "Model should have sigmoid layer"


def test_conv1_has_correct_input_output_channels():
    """
    Test that first convolutional layer has correct input/output channels.
    **Validates: Requirements 3.2**
    """
    model = OrderBookCNN()
    
    # First conv layer: 1 input channel (grayscale) -> 16 output channels
    assert model.conv1.in_channels == 1, \
        "First conv layer should have 1 input channel (grayscale)"
    assert model.conv1.out_channels == 16, \
        "First conv layer should have 16 output channels"


def test_conv2_has_correct_input_output_channels():
    """
    Test that second convolutional layer has correct input/output channels.
    **Validates: Requirements 3.2**
    """
    model = OrderBookCNN()
    
    # Second conv layer: 16 input channels -> 32 output channels
    assert model.conv2.in_channels == 16, \
        "Second conv layer should have 16 input channels"
    assert model.conv2.out_channels == 32, \
        "Second conv layer should have 32 output channels"


def test_conv1_has_correct_kernel_stride_padding():
    """
    Test that first convolutional layer has correct kernel_size, stride, and padding.
    **Validates: Requirements 3.5**
    """
    model = OrderBookCNN()
    
    # Conv1: kernel_size=3, stride=1, padding=1
    assert model.conv1.kernel_size == (3, 3), \
        "First conv layer should have kernel_size=3"
    assert model.conv1.stride == (1, 1), \
        "First conv layer should have stride=1"
    assert model.conv1.padding == (1, 1), \
        "First conv layer should have padding=1"


def test_conv2_has_correct_kernel_stride_padding():
    """
    Test that second convolutional layer has correct kernel_size, stride, and padding.
    **Validates: Requirements 3.5**
    """
    model = OrderBookCNN()
    
    # Conv2: kernel_size=3, stride=1, padding=1
    assert model.conv2.kernel_size == (3, 3), \
        "Second conv layer should have kernel_size=3"
    assert model.conv2.stride == (1, 1), \
        "Second conv layer should have stride=1"
    assert model.conv2.padding == (1, 1), \
        "Second conv layer should have padding=1"


def test_pool1_has_correct_kernel_and_stride():
    """
    Test that first pooling layer has correct kernel_size and stride.
    **Validates: Requirements 3.6**
    """
    model = OrderBookCNN()
    
    # Pool1: kernel_size=2, stride=2
    assert model.pool1.kernel_size == 2, \
        "First pooling layer should have kernel_size=2"
    assert model.pool1.stride == 2, \
        "First pooling layer should have stride=2"


def test_pool2_has_correct_kernel_and_stride():
    """
    Test that second pooling layer has correct kernel_size and stride.
    **Validates: Requirements 3.6**
    """
    model = OrderBookCNN()
    
    # Pool2: kernel_size=2, stride=2
    assert model.pool2.kernel_size == 2, \
        "Second pooling layer should have kernel_size=2"
    assert model.pool2.stride == 2, \
        "Second pooling layer should have stride=2"


def test_dropout_rate_is_correct():
    """
    Test that dropout layers have the correct dropout rate (0.3).
    **Validates: Requirements 3.7**
    """
    model = OrderBookCNN()
    
    # Both dropout layers should have p=0.3
    assert model.dropout1.p == 0.3, \
        "First dropout layer should have dropout rate of 0.3"
    assert model.dropout2.p == 0.3, \
        "Second dropout layer should have dropout rate of 0.3"


def test_dropout_rate_can_be_customized():
    """
    Test that dropout rate can be customized via constructor parameter.
    **Validates: Requirements 3.7**
    """
    custom_dropout = 0.5
    model = OrderBookCNN(dropout_rate=custom_dropout)
    
    # Both dropout layers should have the custom rate
    assert model.dropout1.p == custom_dropout, \
        f"First dropout layer should have dropout rate of {custom_dropout}"
    assert model.dropout2.p == custom_dropout, \
        f"Second dropout layer should have dropout rate of {custom_dropout}"


def test_final_output_layer_has_one_neuron():
    """
    Test that the final fully connected layer outputs 1 neuron.
    **Validates: Requirements 3.2**
    """
    model = OrderBookCNN()
    
    # FC layer should output 1 value (anomaly score)
    assert model.fc.out_features == 1, \
        "Final fully connected layer should have 1 output neuron"


def test_fc_layer_has_correct_input_size():
    """
    Test that the fully connected layer has correct input size (32*16*16=8192).
    **Validates: Requirements 3.2**
    """
    model = OrderBookCNN()
    
    # After two 2x2 max pools: 64 -> 32 -> 16
    # With 32 channels: 32 * 16 * 16 = 8192
    expected_input_size = 32 * 16 * 16
    assert model.fc.in_features == expected_input_size, \
        f"FC layer should have {expected_input_size} input features"


def test_model_forward_pass_with_correct_input_shape():
    """
    Test that model can process input of shape (batch_size, 1, 64, 64).
    **Validates: Requirements 3.2**
    """
    model = OrderBookCNN()
    model.eval()
    
    # Create a batch of 4 heatmaps
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 64, 64)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Output should have shape (batch_size, 1)
    assert output.shape == (batch_size, 1), \
        f"Output shape should be ({batch_size}, 1), got {output.shape}"


def test_model_output_is_in_valid_range():
    """
    Test that model output is in [0, 1] range due to sigmoid activation.
    **Validates: Requirements 3.4**
    """
    model = OrderBookCNN()
    model.eval()
    
    # Create random input
    input_tensor = torch.randn(10, 1, 64, 64)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # All outputs should be in [0, 1]
    assert torch.all(output >= 0.0), \
        "All output values should be >= 0.0"
    assert torch.all(output <= 1.0), \
        "All output values should be <= 1.0"


def test_model_has_correct_layer_types():
    """
    Test that model layers are of the correct types.
    **Validates: Requirements 3.1, 3.2**
    """
    model = OrderBookCNN()
    
    # Check layer types
    assert isinstance(model.conv1, nn.Conv2d), "conv1 should be Conv2d"
    assert isinstance(model.bn1, nn.BatchNorm2d), "bn1 should be BatchNorm2d"
    assert isinstance(model.relu1, nn.ReLU), "relu1 should be ReLU"
    assert isinstance(model.pool1, nn.MaxPool2d), "pool1 should be MaxPool2d"
    assert isinstance(model.dropout1, nn.Dropout), "dropout1 should be Dropout"
    
    assert isinstance(model.conv2, nn.Conv2d), "conv2 should be Conv2d"
    assert isinstance(model.bn2, nn.BatchNorm2d), "bn2 should be BatchNorm2d"
    assert isinstance(model.relu2, nn.ReLU), "relu2 should be ReLU"
    assert isinstance(model.pool2, nn.MaxPool2d), "pool2 should be MaxPool2d"
    assert isinstance(model.dropout2, nn.Dropout), "dropout2 should be Dropout"
    
    assert isinstance(model.flatten, nn.Flatten), "flatten should be Flatten"
    assert isinstance(model.fc, nn.Linear), "fc should be Linear"
    assert isinstance(model.sigmoid, nn.Sigmoid), "sigmoid should be Sigmoid"


def test_batchnorm_layers_have_correct_num_features():
    """
    Test that BatchNorm layers have correct number of features.
    **Validates: Requirements 3.2**
    """
    model = OrderBookCNN()
    
    # bn1 should normalize 16 channels
    assert model.bn1.num_features == 16, \
        "First BatchNorm layer should have 16 features"
    
    # bn2 should normalize 32 channels
    assert model.bn2.num_features == 32, \
        "Second BatchNorm layer should have 32 features"
