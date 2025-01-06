import pytest
from unittest import mock
import pandas as pd
import sys
import os

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

# Now you can import KeywordFeatureExtractorBoxPlots, BoxPlotsVisualizer from modules
from modules import KeywordFeatureExtractorBoxPlots, BoxPlotsVisualizer

@pytest.fixture(autouse=True)
def mock_makedirs():
    with mock.patch('os.makedirs') as mocked_makedirs:
        yield mocked_makedirs

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'description': [
            'This is a software engineer job with Python and Java skills required.',
            'Looking for a data scientist with expertise in machine learning and Python.',
            'Marketing role with experience in digital marketing and SEO.',
            'Product manager with experience in agile and project management.'
        ]
    })

@pytest.fixture
def keyword_dict():
    return {
        'Software Engineer': ['Python', 'Java'],
        'Data Scientist': ['machine learning', 'Python'],
        'Marketing': ['digital marketing', 'SEO'],
        'Product Manager': ['agile', 'project management']
    }

@pytest.fixture
def trend_df():
    return pd.DataFrame({
        'year_month': pd.date_range(start='2020-01-01', periods=6, freq='ME'),
        'role1': [10, 20, 30, 40, 50, 60],
        'role2': [5, 15, 25, 35, 45, 55],
        'role3': [0, 10, 20, 30, 40, 50]
    })

@pytest.fixture
def monthly_trend_df():
    return pd.DataFrame({
        'month': [1, 2, 3, 4, 5, 6],
        'role1_monthly': [5, 10, 15, 20, 25, 30],
        'role2_monthly': [2, 4, 6, 8, 10, 12],
        'role3_monthly': [0, 2, 4, 6, 8, 10]
    })

def test_keyword_feature_extractor_init(sample_df, keyword_dict):
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_df, column='description', keyword_dict=keyword_dict
    )
    assert extractor.df.shape == (4, 1)
    assert extractor.column == 'description'
    assert extractor.keyword_dict == keyword_dict

def test_extract_features(sample_df, keyword_dict):
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_df, column='description', keyword_dict=keyword_dict
    )
    feature_df = extractor.extract_features()
    assert 'Software Engineer' in feature_df.columns
    assert 'Data Scientist' in feature_df.columns
    assert 'Marketing' in feature_df.columns
    assert 'Product Manager' in feature_df.columns
    assert 'Other' in feature_df.columns
    # Adjust to actual shape if additional columns are expected
    assert feature_df.shape == (4, len(keyword_dict) + 2)

def test_box_plots_visualizer_init(trend_df, monthly_trend_df):
    visualizer = BoxPlotsVisualizer(
        trend_df=trend_df, 
        monthly_trend_df=monthly_trend_df, 
        role_columns=['role1', 'role2', 'role3'], 
        output_subfolder_base='box_plots', 
        reports_folder_path='./reports', 
        name_of_topics='Role'
    )
    assert visualizer.trend_df.shape == (6, 4)
    assert visualizer.monthly_trend_df.shape == (6, 4)
    assert visualizer.role_columns == ['role1', 'role2', 'role3']

def test_plot_distribution(trend_df, monthly_trend_df):
    visualizer = BoxPlotsVisualizer(
        trend_df=trend_df, 
        monthly_trend_df=monthly_trend_df, 
        role_columns=['role1', 'role2', 'role3'], 
        output_subfolder_base='box_plots', 
        reports_folder_path='./reports', 
        name_of_topics='Role'
    )
    
    # Mocking os.makedirs and plt.savefig to avoid file creation during test
    with mock.patch('os.makedirs') as mock_makedirs, mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        visualizer.plot_distribution()
        mock_makedirs.assert_called()  # Ensure makedirs is called
        mock_savefig.assert_called()  # Ensure savefig is called
        assert mock_savefig.call_count == 2  # Two plots are generated: by median and mean

def test_box_plots_visualizer_invalid_data():
    # Testing with invalid trend data
    invalid_trend_df = pd.DataFrame({
        'year_month': pd.date_range(start='2020-01-01', periods=6, freq='ME'),
        'role1': [10, 20, 30, 40, 50, 'invalid'],  # Invalid data (string instead of int/float)
    })
    
    visualizer = BoxPlotsVisualizer(
        trend_df=invalid_trend_df, 
        monthly_trend_df=pd.DataFrame(), 
        role_columns=['role1'], 
        output_subfolder_base='box_plots', 
        reports_folder_path='./reports', 
        name_of_topics='Role'
    )
    
    with pytest.raises(Exception):
        visualizer.plot_distribution()  # Expecting exception due to invalid data

def test_extract_features_column_not_found():
    extractor = KeywordFeatureExtractorBoxPlots(
        df=pd.DataFrame({'text': ['sample text']}), column='invalid_column', keyword_dict={}
    )
    with pytest.raises(SystemExit):  # Should exit due to missing column
        extractor.extract_features()

# Integration Tests
def test_integration_keyword_extractor_and_visualizer(sample_df, keyword_dict, trend_df, monthly_trend_df):
    # Simulating the complete workflow of extracting features and visualizing box plots
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_df, column='description', keyword_dict=keyword_dict
    )
    feature_df = extractor.extract_features()

    # Ensure the feature extraction works
    assert 'Software Engineer' in feature_df.columns
    assert 'Other' in feature_df.columns

    # Now use the extracted features in BoxPlotsVisualizer
    visualizer = BoxPlotsVisualizer(
        trend_df=trend_df,
        monthly_trend_df=monthly_trend_df,
        role_columns=['role1', 'role2', 'role3'],
        output_subfolder_base='box_plots',
        reports_folder_path='./reports',
        name_of_topics='Role'
    )

    with mock.patch('os.makedirs') as mock_makedirs, mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        visualizer.plot_distribution()
        mock_makedirs.assert_called()
        mock_savefig.assert_called()

def test_integration_with_invalid_data_in_visualizer():
    # Simulate invalid data for the entire flow
    invalid_trend_df = pd.DataFrame({
        'year_month': pd.date_range(start='2020-01-01', periods=6, freq='ME'),
        'role1': [10, 20, 30, 40, 50, 'invalid'],  # Invalid data type
    })
    
    extractor = KeywordFeatureExtractorBoxPlots(
        df=pd.DataFrame({'description': ['sample text']}), column='description', keyword_dict={}
    )
    
    with pytest.raises(Exception):
        visualizer = BoxPlotsVisualizer(
            trend_df=invalid_trend_df, 
            monthly_trend_df=pd.DataFrame(), 
            role_columns=['role1'], 
            output_subfolder_base='box_plots', 
            reports_folder_path='./reports', 
            name_of_topics='Role'
        )
        visualizer.plot_distribution()