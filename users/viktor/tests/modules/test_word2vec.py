import pytest
from unittest import mock
import pandas as pd
import sys
import os

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))
sys.path.insert(0, src_dir)

# Import modules for testing
from modules.box_plots import KeywordFeatureExtractorBoxPlots

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'text': [
            'Analyze sales data with Excel and SQL skills.',
            'Build machine learning models using Python and R.',
            'Design marketing strategies focusing on branding and social media.',
            'Manage projects using agile methodologies and Jira.'
        ]
    })

@pytest.fixture
def keyword_mapping():
    return {
        'Data Analyst': ['Excel', 'SQL'],
        'Data Scientist': ['Python', 'R'],
        'Marketing': ['branding', 'social media'],
        'Project Manager': ['agile', 'Jira']
    }

@pytest.fixture
def trend_data():
    return pd.DataFrame({
        'month': pd.date_range(start='2021-01-01', periods=6, freq='ME'),
        'role_a': [15, 25, 35, 45, 55, 65],
        'role_b': [10, 20, 30, 40, 50, 60],
        'role_c': [5, 15, 25, 35, 45, 55]
    })

@pytest.fixture
def monthly_data():
    return pd.DataFrame({
        'month': [1, 2, 3, 4, 5, 6],
        'role_a_monthly': [5, 10, 15, 20, 25, 30],
        'role_b_monthly': [3, 6, 9, 12, 15, 18],
        'role_c_monthly': [1, 2, 3, 4, 5, 6]
    })

def test_extractor_initialization(sample_data, keyword_mapping):
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_data, column='text', keyword_dict=keyword_mapping
    )
    assert extractor.df.shape == (4, 1)
    assert extractor.column == 'text'
    assert extractor.keyword_dict == keyword_mapping

def test_extract_features_valid(sample_data, keyword_mapping):
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_data, column='text', keyword_dict=keyword_mapping
    )
    features = extractor.extract_features()
    assert all(key in features.columns for key in keyword_mapping.keys())
    assert 'Other' in features.columns
    assert features.shape[0] == sample_data.shape[0]

def test_extractor_column_not_found():
    with pytest.raises(SystemExit):
        KeywordFeatureExtractorBoxPlots(
            df=pd.DataFrame({'info': ['missing column']}), 
            column='text', 
            keyword_dict={}
        ).extract_features()

def test_extract_single_row_feature(sample_data, keyword_mapping):
    """Unit test: Verify that feature extraction works for a single row of data."""
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_data.head(1), column='text', keyword_dict=keyword_mapping
    )
    features = extractor.extract_features()

    # Check that features for each keyword are correctly extracted for a single row
    assert features.shape[0] == 1  # Only one row
    assert 'Data Analyst' in features.columns
    assert 'Data Scientist' in features.columns
    assert 'Marketing' in features.columns
    assert 'Project Manager' in features.columns

def test_extract_features_includes_other_column(sample_data, keyword_mapping):
    """Unit test: Verify that the 'Other' column is included in the extracted features."""
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_data, column='text', keyword_dict=keyword_mapping
    )
    features = extractor.extract_features()

    assert 'Other' in features.columns  # Ensure the 'Other' column is included
    assert features['Other'].sum() > 0  # There should be some non-zero values in 'Other'

def test_custom_keyword_mapping(sample_data):
    """Unit test: Verify that custom keyword mapping works correctly."""
    custom_mapping = {
        'Developer': ['Python', 'Java'],
        'Designer': ['UI', 'UX'],
    }
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_data, column='text', keyword_dict=custom_mapping
    )
    features = extractor.extract_features()

    # Check if the custom roles are correctly extracted
    assert 'Developer' in features.columns
    assert 'Designer' in features.columns
    assert features['Developer'].sum() > 0  # Some rows should match 'Python' or 'Java'
    assert features['Designer'].sum() > 0  # Some rows should match 'UI' or 'UX'

# Integration Tests
def test_integration_keyword_extraction(sample_data, keyword_mapping):
    """Integration test: Verify that the feature extraction process works with real data."""
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_data, column='text', keyword_dict=keyword_mapping
    )
    features = extractor.extract_features()

    # Use pytest.approx to handle floating point comparisons
    assert features['Data Analyst'].sum() == pytest.approx(1, rel=0.1)  # Allow some margin for floating point error
    assert features['Data Scientist'].sum() == pytest.approx(1, rel=0.1)
    assert features['Marketing'].sum() == pytest.approx(1, rel=0.1)
    assert features['Project Manager'].sum() == pytest.approx(1, rel=0.1)

def test_integration_trend_analysis(trend_data):
    """Integration test: Verify that trend data is processed correctly."""
    extractor = KeywordFeatureExtractorBoxPlots(
        df=trend_data, column='month', keyword_dict={}
    )

    # Since 'extract_trends()' does not exist, you can either:
    # 1. Remove the trend analysis test or
    # 2. Replace it with a call to an existing method, like 'extract_features'
    
    features = extractor.extract_features()  # Assuming you are still extracting features
    assert features.shape[0] == trend_data.shape[0]  # Ensure number of rows matches
    assert 'role_a' in features.columns  # Check that role_a column exists
    assert features['role_a'].sum() == pytest.approx(sum(trend_data['role_a']), rel=0.1)  # Validate role_a values

def test_integration_real_world_data(sample_data, keyword_mapping):
    """Integration test: Verify that feature extraction works with real-world data."""
    extractor = KeywordFeatureExtractorBoxPlots(
        df=sample_data, column='text', keyword_dict=keyword_mapping
    )
    features = extractor.extract_features()

    # Validate that all expected roles are present in the output features
    for role in keyword_mapping.keys():
        assert role in features.columns

    # Check if the number of rows in features matches the input data
    assert features.shape[0] == sample_data.shape[0]

    # Ensure 'Other' column is included and populated
    assert 'Other' in features.columns
    assert features['Other'].sum() > 0  # Ensure some rows have values in 'Other'

def test_integration_trend_data_processing(trend_data):
    """Integration test: Verify that trend data is processed correctly."""
    extractor = KeywordFeatureExtractorBoxPlots(
        df=trend_data, column='month', keyword_dict={}
    )
    
    # Assuming you are using 'extract_features' method to handle trend data
    features = extractor.extract_features()

    # Check that trend data has been processed and that there are values for roles
    assert 'role_a' in features.columns
    assert 'role_b' in features.columns
    assert 'role_c' in features.columns
    assert features['role_a'].sum() == pytest.approx(sum(trend_data['role_a']), rel=0.1)
    assert features['role_b'].sum() == pytest.approx(sum(trend_data['role_b']), rel=0.1)
    assert features['role_c'].sum() == pytest.approx(sum(trend_data['role_c']), rel=0.1)