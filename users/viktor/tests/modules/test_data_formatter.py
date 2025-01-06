import pytest
import pandas as pd
from unittest import mock
import sys
import os

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

# Now you can import DataFormatter from modules
from modules import DataFormatter

@pytest.fixture
def sample_dataframe():
    data = {
        "Id": [1, 2, 3],
        "Description": ["Desc1", "Desc2", "Desc3"],
        "Title": ["Title1", "Title2", "Title3"],
        "Language": ["English", "French", "Spanish"],
        "CreatedAt": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Compensation": [1000, 2000, 3000],
        "AddressId": ["Loc1", "Loc2", "Loc3"],
    }
    return pd.DataFrame(data)

@pytest.fixture
def formatter_instance(sample_dataframe):
    column_renames = {
        "Id": "job_id",
        "Description": "description",
        "Title": "title",
        "Language": "language",
        "CreatedAt": "original_listed_time",
        "Compensation": "med_salary_monthly",
        "AddressId": "location",
    }
    special_handlings_columns = {"location": "company_name"}
    return DataFormatter(sample_dataframe, column_renames, special_handlings_columns)

def test_rename_columns_all_columns_exist(formatter_instance):
    df = formatter_instance.rename_columns()
    
    # Verify renamed columns
    assert "job_id" in df.columns
    assert "description" in df.columns
    assert "title" in df.columns
    assert "language" in df.columns
    assert "original_listed_time" in df.columns
    assert "med_salary_monthly" in df.columns
    assert "location" in df.columns
    assert "company_name" in df.columns

    # Verify original columns are gone
    assert "Id" not in df.columns
    assert "Description" not in df.columns
    assert "Title" not in df.columns

    # Verify company_name matches location
    assert (df["company_name"] == df["location"]).all()

def test_rename_columns_missing_columns():
    # Create a DataFrame missing some columns
    data = {
        "Id": [1, 2, 3],
        "Description": ["Desc1", "Desc2", "Desc3"],
    }
    df = pd.DataFrame(data)
    column_renames = {
        "Id": "job_id",
        "Description": "description",
        "Title": "title",  # Missing in the DataFrame
    }
    special_handlings_columns = {"location": "company_name"}  # Missing in the DataFrame

    formatter = DataFormatter(df, column_renames, special_handlings_columns)
    df = formatter.rename_columns()

    # Verify renamed columns
    assert "job_id" in df.columns
    assert "description" in df.columns

    # Verify missing columns were handled gracefully
    assert "title" not in df.columns
    assert "company_name" not in df.columns

def test_rename_columns_empty_dataframe():
    df = pd.DataFrame()
    column_renames = {
        "Id": "job_id",
        "Description": "description",
    }
    special_handlings_columns = {"location": "company_name"}

    formatter = DataFormatter(df, column_renames, special_handlings_columns)
    df = formatter.rename_columns()

    # Verify DataFrame remains empty
    assert df.empty

def test_integration_with_mocked_framework():
    mock_repo = mock.Mock()
    sample_data = {
        "Id": [1],
        "Description": ["Sample"],
        "AddressId": ["SampleLocation"],
    }
    mock_repo.get_data.return_value = pd.DataFrame(sample_data)

    column_renames = {
        "Id": "job_id",
        "Description": "description",
        "AddressId": "location",
    }
    special_handlings_columns = {"location": "company_name"}

    # Mock getting data from repository
    df = mock_repo.get_data()
    formatter = DataFormatter(df, column_renames, special_handlings_columns)
    formatted_df = formatter.rename_columns()

    # Validate formatted DataFrame
    assert "job_id" in formatted_df.columns
    assert "description" in formatted_df.columns
    assert "location" in formatted_df.columns
    assert "company_name" in formatted_df.columns

    # Verify interactions with the mock
    mock_repo.get_data.assert_called_once()