import pytest
from unittest import mock
from unittest.mock import patch
from nltk.tokenize import word_tokenize

import os
import sys

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

from modules import TextPreprocessor

@pytest.fixture
def sample_texts():
    return [
        "This is a sample text with some stopwords.",
        "Another example of text preprocessing!",
        "Yet another line, testing the tokenizer."
    ]

@pytest.fixture
def stopword_files(tmp_path):
    stopword_file = tmp_path / "stopwords.txt"
    stopword_file.write_text("is\nwith\nand\nanother\nof\nthe\nthis\na\nthe\nthat\nand\nwill\nbe")
    return [str(stopword_file)]

# Unit Tests
def test_load_stopwords(stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    expected_stopwords = {"is", "with", "and", "another", "of", "the", "this", "a", "the", "that", "and", "will", "be"}
    assert preprocessor.stopwords == expected_stopwords

def test_preprocess(sample_texts, stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    processed = preprocessor.preprocess(sample_texts)
    expected = [
        "sample text some stopwords",
        "example text preprocessing",
        "yet line testing tokenizer"
    ]
    assert processed == expected

def test_preprocess_with_punctuation(stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    sample_texts = ["Hello, world! This is a test."]
    processed = preprocessor.preprocess(sample_texts)
    expected = ["hello world test"]
    assert processed == expected

def test_preprocess_with_non_alphabetic_tokens(stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    sample_texts = ["The price is 100 dollars.", "He scored 3 goals!"]
    processed = preprocessor.preprocess(sample_texts)
    expected = ["price dollars", "he scored goals"]
    assert processed == expected

def test_preprocess_with_all_stopwords(stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    sample_texts = ["is with and another of the"]
    processed = preprocessor.preprocess(sample_texts)
    expected = [""]
    assert processed == expected

# Integration Tests
def test_integration_preprocess(sample_texts, stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    processed = preprocessor.preprocess(sample_texts)
    expected = [
        "sample text some stopwords",
        "example text preprocessing",
        "yet line testing tokenizer"
    ]
    assert processed == expected

def test_integration_with_long_text(stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    sample_texts = [
        "This is a longer text that contains several words, including stopwords, and more text.",
        "Another example with even more content, ensuring that the preprocessing works correctly!"
    ]
    processed = preprocessor.preprocess(sample_texts)
    expected = [
        "longer text contains several words including stopwords more text",
        "example even more content ensuring preprocessing works correctly"
    ]
    assert processed == expected

def test_integration_with_mixed_case(stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    sample_texts = ["ThIs Is A MiXeD CaSe TeSt."]
    processed = preprocessor.preprocess(sample_texts)
    expected = ["mixed case test"]
    assert processed == expected

def test_integration_with_multiple_stopword_files(tmp_path):
    # Create two stopword files
    stopword_file_1 = tmp_path / "stopwords_1.txt"
    stopword_file_1.write_text("is\nwith\nand\nanother\nfrom")
    stopword_file_2 = tmp_path / "stopwords_2.txt"
    stopword_file_2.write_text("the\nof\nfor\nof\nthe\nthis\na\nthat")

    stopword_files = [str(stopword_file_1), str(stopword_file_2)]
    preprocessor = TextPreprocessor(stopword_files)
    sample_texts = ["This is a test with multiple stopwords from two files."]
    processed = preprocessor.preprocess(sample_texts)
    expected = ["test multiple stopwords two files"]
    assert processed == expected

def test_integration_with_numbers(stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    sample_texts = ["The year 2025 will be a milestone.", "Someone has 100 dollars."]
    processed = preprocessor.preprocess(sample_texts)
    expected = ["year milestone", "someone has dollars"]
    assert processed == expected

# Edge Case Tests
def test_empty_text_list(stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    processed = preprocessor.preprocess([])
    assert processed == []

def test_text_with_only_stopwords(stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    processed = preprocessor.preprocess(["is with and another"])
    assert processed == [""]

@patch("os.makedirs")
def test_integration_no_folder_creation(mock_makedirs, sample_texts, stopword_files):
    preprocessor = TextPreprocessor(stopword_files)
    processed = preprocessor.preprocess(sample_texts)
    assert mock_makedirs.call_count == 0  # Ensure no folder creation is attempted
    assert len(processed) == len(sample_texts)