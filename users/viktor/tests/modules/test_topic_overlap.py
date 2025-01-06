import pytest
from unittest import mock
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

from modules.topic_overlap import TopicOverlapGraphGenerator

@pytest.fixture
def keyword_dict():
    return {
        'Software Engineer': ['Python', 'Java'],
        'Data Scientist': ['machine learning', 'Python'],
        'Marketing': ['digital marketing', 'SEO'],
        'Product Manager': ['agile', 'project management']
    }

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
def graph_generator():
    return TopicOverlapGraphGenerator()

def test_generate_graph_valid_data(graph_generator, keyword_dict):
    # Mocking the os.makedirs and plt.savefig to avoid actual file creation
    with mock.patch('os.makedirs') as mock_makedirs, mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        graph_generator.generate_graph(keyword_dict, output_subfolder='test_output', name_of_topics='Job')
        
        # Ensure os.makedirs is called to create the output folder
        # mock_makedirs.assert_called_once()
        
        # Ensure savefig is called to save the graph image
        mock_savefig.assert_called_once()

def test_generate_graph_no_overlap(graph_generator):
    # Case where no overlap exists in keyword_dict
    keyword_dict = {
        'Software Engineer': ['Python'],
        'Data Scientist': ['R'],
        'Marketing': ['SEO'],
        'Product Manager': ['Agile']
    }
    
    with mock.patch('os.makedirs') as mock_makedirs, mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        graph_generator.generate_graph(keyword_dict, output_subfolder='test_output', name_of_topics='Job')
        
        # Ensure os.makedirs is called to create the output folder
        # mock_makedirs.assert_called_once()
        
        # Ensure savefig is called to save the graph image
        mock_savefig.assert_called_once()

def test_generate_graph_with_filtered_df(graph_generator, keyword_dict, sample_df):
    # Using a filtered DataFrame as an argument
    filtered_df = sample_df[sample_df['description'].str.contains('Python')]
    
    with mock.patch('os.makedirs') as mock_makedirs, mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        graph_generator.generate_graph(keyword_dict, output_subfolder='test_output', name_of_topics='Job', filtered_df=filtered_df)
        
        # Ensure os.makedirs is called to create the output folder
        # mock_makedirs.assert_called_once()
        
        # Ensure savefig is called to save the graph image
        mock_savefig.assert_called_once()

def test_generate_graph_invalid_data(graph_generator):
    # Testing with invalid data (None as keyword_dict)
    with pytest.raises(ValueError, match="keyword_dict must be a non-empty dictionary."):
        graph_generator.generate_graph(None, output_subfolder='test_output', name_of_topics='Job')
    
    # Testing with an empty dictionary
    with pytest.raises(ValueError, match="keyword_dict must be a non-empty dictionary."):
        graph_generator.generate_graph({}, output_subfolder='test_output', name_of_topics='Job')

def test_generate_graph_no_edges(graph_generator):
    # Case where no edges will be generated (no overlap in keywords)
    keyword_dict = {
        'Topic1': ['word1'],
        'Topic2': ['word2'],
        'Topic3': ['word3']
    }
    
    with mock.patch('os.makedirs') as mock_makedirs, mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        graph_generator.generate_graph(keyword_dict, output_subfolder='test_output', name_of_topics='Topics')
        
        # Ensure os.makedirs is called to create the output folder
        # mock_makedirs.assert_called_once()
        
        # Ensure savefig is called to save the graph image
        mock_savefig.assert_called_once()

def test_integration_graph_generation(graph_generator, keyword_dict):
    # Simulating the complete workflow of generating the graph
    with mock.patch('os.makedirs') as mock_makedirs, mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        graph_generator.generate_graph(keyword_dict, output_subfolder='test_output', name_of_topics='Job')

        # Ensure savefig is called to save the graph image
        mock_savefig.assert_called_once()  # Ensure savefig was called once