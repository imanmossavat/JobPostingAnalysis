import pytest
from unittest import mock
import numpy as np
import os
import sys

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

from modules import SoftmaxWithTemperature

@pytest.fixture
def softmax_transformer():
    return SoftmaxWithTemperature(temp=1.0)

@pytest.fixture
def sample_row():
    return np.array([1.0, 2.0, 3.0, 4.0])

# Unit Tests
def test_softmax_initialization():
    transformer = SoftmaxWithTemperature(temp=2.0)
    assert transformer.temp == 2.0

def test_apply_softmax_with_default_temperature(softmax_transformer, sample_row):
    result = softmax_transformer.apply(sample_row)
    expected = np.exp(sample_row) / np.sum(np.exp(sample_row))  # Standard softmax
    assert np.allclose(result, expected)

def test_apply_softmax_with_custom_temperature(sample_row):
    transformer = SoftmaxWithTemperature(temp=0.5)
    result = transformer.apply(sample_row)
    expected = np.exp(sample_row / 0.5) / np.sum(np.exp(sample_row / 0.5))  # Softmax with temperature
    assert np.allclose(result, expected)

def test_set_temperature(softmax_transformer):
    softmax_transformer.set_temperature(3.0)
    assert softmax_transformer.temp == 3.0

def test_apply_after_temperature_change(softmax_transformer, sample_row):
    softmax_transformer.set_temperature(2.0)
    result = softmax_transformer.apply(sample_row)
    expected = np.exp(sample_row / 2.0) / np.sum(np.exp(sample_row / 2.0))
    assert np.allclose(result, expected)

# Integration Tests
def test_integration_apply_and_set_temperature(sample_row):
    transformer = SoftmaxWithTemperature(temp=1.0)
    initial_result = transformer.apply(sample_row)
    expected_initial = np.exp(sample_row) / np.sum(np.exp(sample_row))
    assert np.allclose(initial_result, expected_initial)

    transformer.set_temperature(0.5)
    new_result = transformer.apply(sample_row)
    expected_new = np.exp(sample_row / 0.5) / np.sum(np.exp(sample_row / 0.5))
    assert np.allclose(new_result, expected_new)

# Additional Integration Tests
def test_integration_apply_with_multiple_rows():
    transformer = SoftmaxWithTemperature(temp=1.0)
    rows = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 0.0, 0.0]])
    results = np.array([transformer.apply(row) for row in rows])
    
    expected_results = np.array([np.exp(row) / np.sum(np.exp(row)) for row in rows])
    assert np.allclose(results, expected_results)

def test_integration_with_extreme_temperature_high(sample_row):
    transformer = SoftmaxWithTemperature(temp=100.0)  # Very high temperature
    result = transformer.apply(sample_row)
    expected = np.exp(sample_row / 100.0) / np.sum(np.exp(sample_row / 100.0))
    assert np.allclose(result, expected)

def test_integration_with_extreme_temperature_low(sample_row):
    transformer = SoftmaxWithTemperature(temp=0.01)  # Very low temperature
    result = transformer.apply(sample_row)
    expected = np.exp(sample_row / 0.01) / np.sum(np.exp(sample_row / 0.01))
    assert np.allclose(result, expected)

def test_integration_chaining_methods(sample_row):
    transformer = SoftmaxWithTemperature(temp=1.0)
    result1 = transformer.apply(sample_row)  # Apply with initial temp
    transformer.set_temperature(0.5)  # Change temperature
    result2 = transformer.apply(sample_row)  # Apply with new temp
    
    expected1 = np.exp(sample_row) / np.sum(np.exp(sample_row))
    expected2 = np.exp(sample_row / 0.5) / np.sum(np.exp(sample_row / 0.5))
    
    assert np.allclose(result1, expected1)
    assert np.allclose(result2, expected2)

def test_integration_with_edge_cases():
    transformer = SoftmaxWithTemperature(temp=1.0)
    
    # Edge case 1: All zeros
    row_zeros = np.array([0.0, 0.0, 0.0])
    result_zeros = transformer.apply(row_zeros)
    expected_zeros = np.array([1/3, 1/3, 1/3])  # Equal probabilities
    assert np.allclose(result_zeros, expected_zeros)
    
    # Edge case 2: Large numbers
    row_large = np.array([1000.0, 1000.1, 1000.2])
    result_large = transformer.apply(row_large)
    expected_large = np.exp(row_large - np.max(row_large)) / np.sum(np.exp(row_large - np.max(row_large)))
    assert np.allclose(result_large, expected_large)

def test_integration_temperature_effect_consistency(sample_row):
    transformer = SoftmaxWithTemperature(temp=1.0)
    result1 = transformer.apply(sample_row)
    transformer.set_temperature(2.0)
    result2 = transformer.apply(sample_row)
    transformer.set_temperature(1.0)
    result3 = transformer.apply(sample_row)
    
    expected1 = np.exp(sample_row) / np.sum(np.exp(sample_row))
    expected2 = np.exp(sample_row / 2.0) / np.sum(np.exp(sample_row / 2.0))
    expected3 = expected1  # Should match the first result since temperature is reset
    
    assert np.allclose(result1, expected1)
    assert np.allclose(result2, expected2)
    assert np.allclose(result3, expected3)

# Mock Tests for External Dependencies
def test_mocking_temperature_setting():
    with mock.patch.object(SoftmaxWithTemperature, 'set_temperature', return_value=None) as mock_set_temp:
        transformer = SoftmaxWithTemperature(temp=1.0)
        transformer.set_temperature(2.0)
        mock_set_temp.assert_called_once_with(2.0)

def test_mocking_apply_method(sample_row):
    with mock.patch.object(SoftmaxWithTemperature, 'apply', return_value=np.array([0.1, 0.2, 0.3, 0.4])) as mock_apply:
        transformer = SoftmaxWithTemperature(temp=1.0)
        result = transformer.apply(sample_row)
        mock_apply.assert_called_once_with(sample_row)
        assert np.allclose(result, [0.1, 0.2, 0.3, 0.4])