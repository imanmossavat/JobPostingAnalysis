import pytest
from unittest import mock
import pandas as pd
import re
import os
import sys

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

from modules.topic_assigner import TopicAssigner

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

# Unit test for the TopicAssigner class initialization
def test_topic_assigner_init(sample_df, keyword_dict):
    assigner = TopicAssigner()
    assert isinstance(assigner, TopicAssigner)

# Unit test for assigning the most likely topic based on keyword matching
def test_assign_most_likely_topic(sample_df, keyword_dict):
    assigner = TopicAssigner()

    result_df = assigner.assign_most_likely_topic(sample_df, 'description', keyword_dict)

    # Check that the result DataFrame has the new 'Most_Likely_Topic' column
    assert 'Most_Likely_Topic' in result_df.columns

    # Verify that topics are assigned correctly
    assert result_df['Most_Likely_Topic'].iloc[0] == 'Software Engineer'
    assert result_df['Most_Likely_Topic'].iloc[1] == 'Data Scientist'
    assert result_df['Most_Likely_Topic'].iloc[2] == 'Marketing'
    assert result_df['Most_Likely_Topic'].iloc[3] == 'Product Manager'

# Unit test for the case when no keywords match (should return 'Unclassified')
def test_assign_unclassified_topic(sample_df, keyword_dict):
    # Modify the sample DataFrame so that none of the descriptions match the keywords
    sample_df['description'] = ['This is a completely unrelated text'] * 4

    assigner = TopicAssigner()

    result_df = assigner.assign_most_likely_topic(sample_df, 'description', keyword_dict)

    # Verify that all rows are assigned 'Unclassified'
    assert all(result_df['Most_Likely_Topic'] == 'Unclassified')

def test_case_insensitive_keyword_matching(sample_df, keyword_dict):
    # Add a description with mixed case for Python
    sample_df['description'] = [
        'This is a software engineer job with Python and Java skills required.',
        'Looking for a data scientist with expertise in machine learning and Python.',
        'Marketing role with experience in digital marketing and SEO.',
        'Product manager with experience in agile and project management.',
    ]

    assigner = TopicAssigner()

    # Assign topics
    result_df = assigner.assign_most_likely_topic(sample_df, 'description', keyword_dict)

    # Assert that the topic is correctly assigned based on case-insensitive keyword matching
    assert result_df['Most_Likely_Topic'][0] == 'Software Engineer'  # Case insensitive matching for 'Python' and 'Java'
    assert result_df['Most_Likely_Topic'][1] == 'Data Scientist'  # Case insensitive matching for 'machine learning' and 'Python'
    assert result_df['Most_Likely_Topic'][2] == 'Marketing'  # Case insensitive matching for 'digital marketing' and 'SEO'
    assert result_df['Most_Likely_Topic'][3] == 'Product Manager'  # Case insensitive matching for 'agile' and 'project management'

# Unit test for invalid column name
def test_column_not_found():
    assigner = TopicAssigner()

    with pytest.raises(KeyError):  # Expecting KeyError for missing column
        assigner.assign_most_likely_topic(pd.DataFrame({'text': ['sample text']}), 'invalid_column', {})

# Unit test for empty DataFrame
def test_empty_dataframe():
    assigner = TopicAssigner()
    
    result_df = assigner.assign_most_likely_topic(pd.DataFrame(columns=['description']), 'description', {})
    
    # Expect an empty DataFrame with the 'Most_Likely_Topic' column
    assert result_df.empty
    assert 'Most_Likely_Topic' in result_df.columns

# Integration Test for full workflow (feature extraction + topic assignment)
def test_integration_topic_assigner(sample_df, keyword_dict):
    assigner = TopicAssigner()

    # Apply the topic assignment
    result_df = assigner.assign_most_likely_topic(sample_df, 'description', keyword_dict)

    # Check that the topic assignment works end-to-end
    assert 'Most_Likely_Topic' in result_df.columns
    assert result_df['Most_Likely_Topic'].iloc[0] == 'Software Engineer'
    assert result_df['Most_Likely_Topic'].iloc[1] == 'Data Scientist'

# Integration test with invalid data (empty or invalid descriptions)
def test_integration_invalid_data(sample_df, keyword_dict):
    # Introduce invalid data (empty descriptions)
    sample_df['description'] = [None, None, None, None]

    assigner = TopicAssigner()

    result_df = assigner.assign_most_likely_topic(sample_df, 'description', keyword_dict)

    # Verify that the result is 'Unclassified' for all rows
    assert all(result_df['Most_Likely_Topic'] == 'Unclassified')

# Integration test with mocking os.makedirs (no new folders should be created)
@mock.patch('os.makedirs')
def test_no_new_folders_created(mock_makedirs, sample_df, keyword_dict):
    assigner = TopicAssigner()

    # Run the topic assignment method
    result_df = assigner.assign_most_likely_topic(sample_df, 'description', keyword_dict)

    # Check that os.makedirs was not called during the process
    mock_makedirs.assert_not_called()