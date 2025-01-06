import pytest
from unittest import mock
import pandas as pd
import os
import sys
from unittest.mock import patch

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

from modules.word_clouds import WordCloudGenerator

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Description': [
            'Software engineer with expertise in Python and Java.',
            'Data scientist experienced in machine learning and Python.',
            'Digital marketing role with SEO skills.',
            'Product manager with experience in agile methodologies.'
        ]
    })

@pytest.fixture
def keyword_dict():
    return {
        'Software Engineer': ['Python', 'Java'],
        'Data Scientist': ['machine learning', 'Python'],
        'Marketing': ['SEO', 'digital marketing'],
        'Product Manager': ['agile', 'project management']
    }

# Mock stopword files for testing
@pytest.fixture
def mock_stopwords():
    return ['the', 'and', 'is', 'in', 'of', 'with']

# Unit Tests

@patch('modules.text_preprocessor.TextPreprocessor.load_stopwords')
@patch('os.makedirs')  # Mock os.makedirs to prevent folder creation
@patch('matplotlib.pyplot.savefig')  # Mock plt.savefig to prevent image saving
def test_wordcloud_generator_init(mock_savefig, mock_makedirs, mock_load_stopwords, sample_df, keyword_dict, mock_stopwords):
    # Mock the stopwords to return a predefined list
    mock_load_stopwords.return_value = mock_stopwords

    generator = WordCloudGenerator(
        df=sample_df,
        keyword_dict=keyword_dict,
        output_folder='./wordclouds',
        name_of_topics='Job Roles',
        stopword_files=['stopwords.txt']
    )
    assert generator.df.shape == (4, 1)
    assert generator.keyword_dict == keyword_dict
    assert generator.output_folder == './wordclouds'
    assert generator.name_of_topics == 'Job Roles'
    assert generator.stopword_files == ['stopwords.txt']

@patch('modules.text_preprocessor.TextPreprocessor.load_stopwords')
@patch('os.makedirs')  # Mock os.makedirs to prevent folder creation
@patch('matplotlib.pyplot.savefig')  # Mock plt.savefig to prevent image saving
def test_generate_wordcloud_for_topic_empty_df(mock_savefig, mock_makedirs, mock_load_stopwords):
    mock_load_stopwords.return_value = ['the', 'and', 'is', 'in', 'of', 'with']
    empty_df = pd.DataFrame({'Description': []})
    generator = WordCloudGenerator(
        df=empty_df,
        keyword_dict={'Software Engineer': ['Python']},
        output_folder='./wordclouds',
        name_of_topics='Job Roles',
        stopword_files=['stopwords.txt']
    )
    result = generator.generate_wordcloud_for_topic()
    assert result == []

@patch('modules.text_preprocessor.TextPreprocessor.load_stopwords')
@patch('os.makedirs')  # Mock os.makedirs to prevent folder creation
@patch('matplotlib.pyplot.savefig')  # Mock plt.savefig to prevent image saving
def test_generate_wordcloud_for_topic_no_keywords(mock_savefig, mock_makedirs, mock_load_stopwords, sample_df, keyword_dict):
    mock_load_stopwords.return_value = ['the', 'and', 'is', 'in', 'of', 'with']
    # Providing an empty keyword list for a topic
    keyword_dict_empty = {'Software Engineer': []}
    generator = WordCloudGenerator(
        df=sample_df,
        keyword_dict=keyword_dict_empty,
        output_folder='./wordclouds',
        name_of_topics='Job Roles',
        stopword_files=['stopwords.txt']
    )
    result = generator.generate_wordcloud_for_topic()
    assert result == []

@patch('modules.text_preprocessor.TextPreprocessor.load_stopwords')
@patch('os.makedirs')  # Mock os.makedirs to prevent folder creation
@patch('matplotlib.pyplot.savefig')  # Mock plt.savefig to prevent image saving
def test_generate_wordcloud_for_topic_no_keywords_matched(mock_savefig, mock_makedirs, mock_load_stopwords, sample_df):
    mock_load_stopwords.return_value = ['the', 'and', 'is', 'in', 'of', 'with']
    keyword_dict_no_match = {'Nonexistent Role': ['Ruby', 'PHP']}
    generator = WordCloudGenerator(
        df=sample_df,
        keyword_dict=keyword_dict_no_match,
        output_folder='./wordclouds',
        name_of_topics='Job Roles',
        stopword_files=['stopwords.txt']
    )

    # Mocking os.makedirs and plt.savefig to avoid file creation during the test
    image_paths = generator.generate_wordcloud_for_topic()
    mock_makedirs.assert_called()
    mock_savefig.assert_not_called()  # No image should be saved since no keywords match
    assert image_paths == []

@patch('modules.text_preprocessor.TextPreprocessor.load_stopwords')
@patch('os.makedirs')  # Mock os.makedirs to prevent folder creation
@patch('matplotlib.pyplot.savefig')  # Mock plt.savefig to prevent image saving
def test_generate_wordcloud_for_topic_with_invalid_column(mock_savefig, mock_makedirs, mock_load_stopwords):
    mock_load_stopwords.return_value = ['the', 'and', 'is', 'in', 'of', 'with']
    invalid_df = pd.DataFrame({'InvalidColumn': ['Text']})
    generator = WordCloudGenerator(
        df=invalid_df,
        keyword_dict={'Software Engineer': ['Python']},
        output_folder='./wordclouds',
        name_of_topics='Job Roles',
        stopword_files=['stopwords.txt'],
        column='Description'  # Non-existent column
    )
    with pytest.raises(KeyError):
        generator.generate_wordcloud_for_topic()

# Integration Tests

@patch('modules.text_preprocessor.TextPreprocessor.load_stopwords')
@patch('os.makedirs')  # Mock os.makedirs to prevent folder creation
@patch('matplotlib.pyplot.savefig')  # Mock plt.savefig to prevent image saving
def test_integration_wordcloud_generator_with_invalid_column(mock_savefig, mock_makedirs, mock_load_stopwords):
    # Mock the stopwords to return a predefined list
    mock_load_stopwords.return_value = ['the', 'and', 'is', 'in', 'of', 'with']

    invalid_df = pd.DataFrame({'InvalidColumn': ['Text']})
    generator = WordCloudGenerator(
        df=invalid_df,
        keyword_dict={'Software Engineer': ['Python']},
        output_folder='./wordclouds',
        name_of_topics='Job Roles',
        stopword_files=['stopwords.txt'],
        column='Description'  # Non-existent column
    )
    with pytest.raises(KeyError):
        generator.generate_wordcloud_for_topic()

@patch('modules.text_preprocessor.TextPreprocessor.load_stopwords')
@patch('os.makedirs')  # Mock os.makedirs to prevent folder creation
@patch('matplotlib.pyplot.savefig')  # Mock plt.savefig to prevent image saving
def test_integration_wordcloud_generator_valid(mock_savefig, mock_makedirs, mock_load_stopwords, sample_df, keyword_dict):
    # Mock the stopwords to return a predefined list
    mock_load_stopwords.return_value = ['the', 'and', 'is', 'in', 'of', 'with']

    # Valid sample dataframe with keywords
    generator = WordCloudGenerator(
        df=sample_df,
        keyword_dict=keyword_dict,
        output_folder='./wordclouds',
        name_of_topics='Job Roles',
        stopword_files=['stopwords.txt']
    )

    # Generate word clouds and check the returned paths
    image_paths = generator.generate_wordcloud_for_topic()

    # Check that the image paths are returned and mock functions were called
    assert len(image_paths) > 0  # Ensure at least one image was generated
    mock_makedirs.assert_called()  # Ensure the output folder creation was attempted
    mock_savefig.assert_called()  # Ensure plt.savefig was called to save the images