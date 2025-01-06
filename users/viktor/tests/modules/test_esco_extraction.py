''' import pytest
import pandas as pd
from unittest import mock
import sys
import os

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))
sys.path.insert(0, src_dir)

from modules.esco_extraction import ESCOAnalyzer, detect_language, initiate_esco_analysis


@pytest.fixture
def sample_texts():
    return [
        {"text": "Python developer with experience in Django.", "lang": "english"},
        {"text": "Développeur Python avec expérience en Django.", "lang": "Other"},
    ]


def test_detect_language_english():
    assert detect_language("Python developer with experience in Django.") == "english"


def test_detect_language_other():
    assert detect_language("Développeur Python avec expérience en Django.") == "Other"


@mock.patch("modules.esco_extraction.pipeline")
def test_extract_skills_and_knowledge(mock_pipeline, sample_texts):
    mock_pipeline.return_value = mock.Mock(
        __call__=lambda text: [
            {"start": 0, "end": 6, "score": 0.99, "word": "Python", "entity_group": "Skill"}
        ]
    )
    analyzer = ESCOAnalyzer()
    analyzer.token_skill_classifier = mock_pipeline.return_value
    analyzer.token_knowledge_classifier = mock_pipeline.return_value

    for text_data in sample_texts:
        result = analyzer.extract_skills_and_knowledge(text_data["text"], text_data["lang"])
        assert "skills" in result
        assert "knowledge" in result
        assert result["detected-language"] == text_data["lang"]


@mock.patch("pandas.DataFrame.to_csv")
def test_save_progress(mock_to_csv):
    analyzer = ESCOAnalyzer()
    sample_data = pd.DataFrame([{"text": "Sample data", "skills": [], "knowledge": []}])
    analyzer.save_progress(sample_data, "/mock/path")

    mock_to_csv.assert_called_once_with(
        "/mock/path/extracted_skills_20240101_120000.csv", index=False, encoding="utf-8-sig"
    )


@mock.patch("modules.esco_extraction.Document")
@mock.patch("modules.esco_extraction.datetime")
@mock.patch("modules.esco_extraction.os.makedirs")
def test_generate_report(mock_makedirs, mock_datetime, mock_document):
    # Mock the current time returned by datetime.now()
    mock_datetime.now.return_value.strftime.return_value = "2024-01-01 12:00:00"

    # Initialize the ESCOAnalyzer instance
    analyzer = ESCOAnalyzer()

    # Call the method under test
    analyzer.generate_report("input.csv", "output.csv", "/mock/reports")

    # Ensure that the directory creation is attempted
    mock_makedirs.assert_any_call("/mock/reports", exist_ok=True)

    # Ensure the save method is called with the correct path
    print(mock_document.return_value.save.call_args_list)  # Debugging output
    mock_document.return_value.save.assert_called_once_with("/mock/reports/experiment_report.docx")


@mock.patch("modules.esco_extraction.ESCOAnalyzer.extract_skills_and_knowledge")
@mock.patch("modules.esco_extraction.ESCOAnalyzer.save_progress")
@mock.patch("modules.esco_extraction.ESCOAnalyzer.generate_report")
@mock.patch("modules.esco_extraction.pd.read_csv")
def test_initiate_esco_analysis(mock_read_csv, mock_generate_report, mock_save_progress, mock_extract_skills):
    # Mock return value for extract_skills_and_knowledge
    mock_extract_skills.return_value = {
        "text": "Sample job description",
        "skills": [{"word": "Python", "entity": "Skill"}],
        "knowledge": [],
        "detected-language": "english",
    }

    # Mock input CSV data
    sample_csv = pd.DataFrame({"Description": ["Sample job description"]})
    mock_read_csv.return_value = sample_csv

    input_file = "/mock/input.csv"
    output_file = "/mock/output.csv"
    output_subfolder = "/mock/output_subfolder"

    # Call the function under test
    initiate_esco_analysis(input_file, output_file, output_subfolder)

    # Validate calls
    mock_read_csv.assert_called_once_with(input_file)
    mock_extract_skills.assert_called()
    mock_save_progress.assert_called()
    mock_generate_report.assert_called_once_with(input_file, output_file, output_subfolder) '''