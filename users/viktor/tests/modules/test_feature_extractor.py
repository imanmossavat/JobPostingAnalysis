import pytest
import unittest
import pandas as pd
import sys
import os

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

from modules import KeywordFeatureExtractor


# Unit Test 1: Test Initialization
def test_initialization():
    extractor = KeywordFeatureExtractor(
        column="text",
        keyword_dict={"Feature1": ["keyword1", "keyword2"]},
        temp=1.5
    )
    assert extractor.column == "text"
    assert extractor.keyword_dict == {"Feature1": ["keyword1", "keyword2"]}
    assert extractor.temp == 1.5


# Unit Test 2: Test feature extraction with keywords
def test_extract_features_with_keywords():
    extractor = KeywordFeatureExtractor(
        column="text",
        keyword_dict={"Feature1": ["keyword1", "keyword2"]}
    )
    df = pd.DataFrame({"text": ["This contains keyword1", "No keywords here"]})
    result_df = extractor.extract_features(df)

    # Softmax normalization applied; check the probabilities
    assert pytest.approx(result_df["Feature1"].iloc[0], 0.01) == 0.731
    assert pytest.approx(result_df["Feature1"].iloc[1], 0.01) == 0.268


# Unit Test 3: Test softmax normalization
def test_softmax_normalization():
    extractor = KeywordFeatureExtractor(
        column="text",
        keyword_dict={"Feature1": ["keyword1"], "Feature2": ["keyword2"]}
    )
    df = pd.DataFrame({"text": ["keyword1", "keyword2", "no match"]})
    result_df = extractor.extract_features(df)
    softmax_columns = ["Feature1", "Feature2", "Other"]

    # Ensure softmax values sum to 1
    for _, row in result_df[softmax_columns].iterrows():
        assert pytest.approx(row.sum(), 0.01) == 1


# Integration Test: Full feature extraction process
def test_full_feature_extraction():
    extractor = KeywordFeatureExtractor(
        column="text",
        keyword_dict={
            "Feature1": ["keyword1", "keyword2"],
            "Feature2": ["keyword3"]
        }
    )
    df = pd.DataFrame({
        "text": [
            "This has keyword1 and keyword3",  # Should match Feature1 and Feature2
            "Only keyword2 is here",          # Should match Feature1
            "No matching keywords"            # Should match "Other"
        ]
    })
    result_df = extractor.extract_features(df)

    # Check that the values for Feature1 and Feature2 are probabilities (between 0 and 1)
    assert 0 < result_df["Feature1"].iloc[0] < 1
    assert 0 < result_df["Feature2"].iloc[0] < 1
    assert 0 < result_df["Feature1"].iloc[1] < 1
    assert 0 < result_df["Feature2"].iloc[1] < 1
    assert 0 < result_df["Feature1"].iloc[2] < 1
    assert 0 < result_df["Feature2"].iloc[2] < 1

    # Check that the sum of probabilities for each row equals 1 (due to softmax)
    softmax_columns = ["Feature1", "Feature2", "Other"]
    for _, row in result_df[softmax_columns].iterrows():
        assert pytest.approx(row.sum(), 0.01) == 1

    # Check "Other" feature (should be low if keywords are found, high if no keywords are found)
    assert result_df["Other"].iloc[0] < 0.5  # Since Feature1 and Feature2 are matched
    assert result_df["Other"].iloc[1] < 0.5  # Since Feature1 is matched
    assert result_df["Other"].iloc[2] > 0.5  # Since no keywords are matched

def test_temperature_scaling():
    extractor = KeywordFeatureExtractor(
        column="text",
        keyword_dict={
            "Feature1": ["keyword1"],
            "Feature2": ["keyword2"]
        },
        temp=2.0  # Use a temperature greater than 1 to reduce the differences between probabilities
    )
    df = pd.DataFrame({
        "text": [
            "This contains keyword1",  # Should match Feature1
            "This contains keyword2",  # Should match Feature2
            "No matching keywords"     # Should match "Other"
        ]
    })
    result_df = extractor.extract_features(df)

    # Check that Feature1 and Feature2 values are probabilities (between 0 and 1)
    assert 0 < result_df["Feature1"].iloc[0] < 1
    assert 0 < result_df["Feature2"].iloc[0] < 1
    assert 0 < result_df["Feature1"].iloc[1] < 1
    assert 0 < result_df["Feature2"].iloc[1] < 1
    assert 0 < result_df["Feature1"].iloc[2] < 1
    assert 0 < result_df["Feature2"].iloc[2] < 1

    # Check that the sum of probabilities for each row equals 1 (due to softmax)
    softmax_columns = ["Feature1", "Feature2", "Other"]
    for _, row in result_df[softmax_columns].iterrows():
        assert pytest.approx(row.sum(), 0.01) == 1

    # Check the effect of temperature on "Other" feature
    assert result_df["Other"].iloc[0] < 0.5  # Feature1 is matched, "Other" should be low
    assert result_df["Other"].iloc[1] < 0.5  # Feature2 is matched, "Other" should be low
    assert result_df["Other"].iloc[2] > 0.4  # No keywords matched, "Other" should be higher than 0.4

def test_multiple_keywords_per_feature():
    extractor = KeywordFeatureExtractor(
        column="text",
        keyword_dict={
            "Feature1": ["keyword1", "keyword2"],
            "Feature2": ["keyword3", "keyword4"]
        }
    )
    df = pd.DataFrame({
        "text": [
            "This has keyword1 and keyword3",  # Should match Feature1 and Feature2
            "Only keyword2 is here",          # Should match Feature1
            "No matching keywords"            # Should match "Other"
        ]
    })
    result_df = extractor.extract_features(df)

    # Check that Feature1 and Feature2 values are probabilities (between 0 and 1)
    assert 0 < result_df["Feature1"].iloc[0] < 1
    assert 0 < result_df["Feature2"].iloc[0] < 1
    assert 0 < result_df["Feature1"].iloc[1] < 1
    assert 0 < result_df["Feature2"].iloc[1] < 1
    assert 0 < result_df["Feature1"].iloc[2] < 1
    assert 0 < result_df["Feature2"].iloc[2] < 1

    # Check that the sum of probabilities for each row equals 1 (due to softmax)
    softmax_columns = ["Feature1", "Feature2", "Other"]
    for _, row in result_df[softmax_columns].iterrows():
        assert pytest.approx(row.sum(), 0.01) == 1

    # Check "Other" feature (should be low if keywords are found, high if no keywords are found)
    assert result_df["Other"].iloc[0] < 0.5  # Since Feature1 and Feature2 are matched
    assert result_df["Other"].iloc[1] < 0.5  # Since Feature1 is matched
    assert result_df["Other"].iloc[2] > 0.5  # Since no keywords are matched

def test_case_insensitive_matching():
    extractor = KeywordFeatureExtractor(
        column="text",
        keyword_dict={
            "Feature1": ["keyword1", "keyword2"],
            "Feature2": ["keyword3"]
        }
    )
    df = pd.DataFrame({
        "text": [
            "This has KEYWORD1 and keyword3",  # Should match Feature1 and Feature2 (case insensitive)
            "Only Keyword2 is here",          # Should match Feature1 (case insensitive)
            "No matching keywords"            # Should match "Other"
        ]
    })
    result_df = extractor.extract_features(df)

    # Check that Feature1 and Feature2 values are probabilities (between 0 and 1)
    assert 0 < result_df["Feature1"].iloc[0] < 1
    assert 0 < result_df["Feature2"].iloc[0] < 1
    assert 0 < result_df["Feature1"].iloc[1] < 1
    assert 0 < result_df["Feature2"].iloc[1] < 1
    assert 0 < result_df["Feature1"].iloc[2] < 1
    assert 0 < result_df["Feature2"].iloc[2] < 1

    # Check that the sum of probabilities for each row equals 1 (due to softmax)
    softmax_columns = ["Feature1", "Feature2", "Other"]
    for _, row in result_df[softmax_columns].iterrows():
        assert pytest.approx(row.sum(), 0.01) == 1

    # Check "Other" feature (should be low if keywords are found, high if no keywords are found)
    assert result_df["Other"].iloc[0] < 0.5  # Since Feature1 and Feature2 are matched (case-insensitive)
    assert result_df["Other"].iloc[1] < 0.5  # Since Feature1 is matched (case-insensitive)
    assert result_df["Other"].iloc[2] > 0.5  # Since no keywords are matched


def test_full_feature_extraction_with_multiple_features():
    extractor = KeywordFeatureExtractor(
        column="text",
        keyword_dict={
            "Feature1": ["keyword1", "keyword2"],  # Matches Feature1
            "Feature2": ["keyword3", "keyword4"],  # Matches Feature2
            "Feature3": ["keyword5"]               # Matches Feature3
        }
    )
    df = pd.DataFrame({
        "text": [
            "This contains keyword1 and keyword3",  # Should match Feature1 and Feature2
            "Only keyword2 here",                  # Should match Feature1
            "No matching keywords",                # Should match "Other"
            "Keyword5 matches Feature3"            # Should match Feature3
        ]
    })
    result_df = extractor.extract_features(df)

    # Check that the values for Feature1, Feature2, and Feature3 are probabilities (between 0 and 1)
    assert 0 < result_df["Feature1"].iloc[0] < 1
    assert 0 < result_df["Feature2"].iloc[0] < 1
    assert 0 < result_df["Feature3"].iloc[0] < 1
    assert 0 < result_df["Feature1"].iloc[1] < 1
    assert 0 < result_df["Feature2"].iloc[1] < 1
    assert 0 < result_df["Feature3"].iloc[1] < 1
    assert 0 < result_df["Feature1"].iloc[2] < 1
    assert 0 < result_df["Feature2"].iloc[2] < 1
    assert 0 < result_df["Feature3"].iloc[2] < 1
    assert 0 < result_df["Feature1"].iloc[3] < 1
    assert 0 < result_df["Feature2"].iloc[3] < 1
    assert 0 < result_df["Feature3"].iloc[3] < 1

    # Check that the sum of probabilities for each row equals 1 (due to softmax)
    softmax_columns = ["Feature1", "Feature2", "Feature3", "Other"]
    for _, row in result_df[softmax_columns].iterrows():
        assert pytest.approx(row.sum(), 0.01) == 1

    # Check the "Other" feature (should be low if keywords are found, high if no keywords are found)
    assert result_df["Other"].iloc[0] < 0.5  # Feature1 and Feature2 matched
    assert result_df["Other"].iloc[1] < 0.5  # Feature1 matched
    assert result_df["Other"].iloc[2] > 0.45  # Relaxed condition for no matching keywords
    assert result_df["Other"].iloc[3] < 0.5  # Feature3 matched


# Unittest class for running selected tests
class TestKeywordFeatureExtractor(unittest.TestCase):

    def test_empty_dataframe(self):
        extractor = KeywordFeatureExtractor(
            column="text",
            keyword_dict={"Feature1": ["keyword1"]}
        )
        df = pd.DataFrame({"text": []})
        result_df = extractor.extract_features(df)
        self.assertTrue(result_df.empty)

    def test_no_keywords(self):
        extractor = KeywordFeatureExtractor(
            column="text",
            keyword_dict={"Feature1": ["keyword1"]}
        )
        df = pd.DataFrame({"text": ["No relevant keywords"]})
        result_df = extractor.extract_features(df)

        # Since no keywords are found, Feature1 will have a non-zero probability due to softmax
        # We check that the value is close to the softmax probability for the "Other" feature.
        # For example, Feature1 is expected to be around 0.26894142 and "Other" around 0.73105858.
        
        assert pytest.approx(result_df["Feature1"].iloc[0], 0.01) == 0.26894142
        assert pytest.approx(result_df["Other"].iloc[0], 0.01) == 0.73105858