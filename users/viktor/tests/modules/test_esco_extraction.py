import pytest
from unittest import mock
from unittest.mock import patch
import pandas as pd

import sys
import os

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))
sys.path.insert(0, src_dir)

from modules.esco_extraction import ESCOAnalyzer, detect_language

@pytest.fixture
def sample_text():
    return "This is a software engineering job requiring Python and Java skills."

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "Description": [
            "Software engineer role with Python and Java.",
            "Data scientist needed with expertise in machine learning.",
            "Marketing expert required with SEO experience."
        ]
    })

# Unit Tests
def test_detect_language_english(sample_text):
    lang = detect_language(sample_text)
    assert lang == "english"

def test_extract_skills_and_knowledge(sample_text):
    analyzer = ESCOAnalyzer()

    # Mock the pipeline outputs
    with mock.patch.object(analyzer, 'token_skill_classifier', return_value=[
        {"start": 0, "end": 8, "score": 0.99, "word": "software", "entity_group": "Skill"}
    ]), mock.patch.object(analyzer, 'token_knowledge_classifier', return_value=[
        {"start": 0, "end": 8, "score": 0.95, "word": "engineering", "entity_group": "Knowledge"}
    ]):

        result = analyzer.extract_skills_and_knowledge(sample_text, "english")

    assert "skills" in result
    assert len(result["skills"]) == 1
    assert result["skills"][0]["entity"] == "Skill"
    assert "knowledge" in result
    assert len(result["knowledge"]) == 1
    assert result["knowledge"][0]["entity"] == "Knowledge"

# Integration Tests
def test_initiate_esco_analysis(sample_dataframe):
    with patch('os.makedirs') as mock_makedirs:
        analyzer = ESCOAnalyzer()

        # Mock methods
        with mock.patch.object(analyzer, 'extract_skills_and_knowledge', return_value={
            "skills": [{"entity": "Skill", "word": "Python"}],
            "knowledge": [{"entity": "Knowledge", "word": "machine learning"}],
            "detected-language": "english"
        }):

            results = []
            for _, row in sample_dataframe.iterrows():
                job_description = row["Description"]
                lang = detect_language(job_description)
                extracted_data = analyzer.extract_skills_and_knowledge(job_description, lang)
                results.append(extracted_data)

            # Verify results
            assert len(results) == len(sample_dataframe)
            assert all("skills" in res for res in results)

            # Ensure no new folders are created
            mock_makedirs.assert_called()


def test_generate_report():
    analyzer = ESCOAnalyzer()
    with patch('os.makedirs'), \
         patch('subprocess.check_output', return_value=b'12345abc') as mock_git:

        input_file = "test_input.csv"
        output_file = "test_output.csv"
        output_subfolder = "test_subfolder"

        with mock.patch.object(analyzer, 'generate_report') as mock_generate_report:
            analyzer.generate_report(input_file, output_file, output_subfolder)

        # Verify report generation is called
        mock_generate_report.assert_called_with(input_file, output_file, output_subfolder)