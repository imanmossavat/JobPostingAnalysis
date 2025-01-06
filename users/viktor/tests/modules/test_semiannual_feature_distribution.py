import pytest
from unittest import mock
import pandas as pd
import os
import sys
from datetime import datetime

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

# Now you can import SemiannualFeatureDistributionPlotter from the appropriate module
from modules import SemiannualFeatureDistributionPlotter

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

@pytest.fixture
def plotter():
    return SemiannualFeatureDistributionPlotter(
        role_columns=['role1', 'role2', 'role3'],
        output_subfolder='./plots',
        name_of_topics='Role'
    )

def test_semiannual_feature_distribution_plotter_init(plotter):
    assert plotter.role_columns == ['role1', 'role2', 'role3']
    assert plotter.output_subfolder == './plots'
    assert plotter.name_of_topics == 'Role'

def test_plot_trends(plotter, trend_df, monthly_trend_df):
    # Mocking os.makedirs and plt.savefig to avoid file creation during the test
    with mock.patch('os.makedirs'), mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        plot_file_path = plotter.plot_trends(trend_df, monthly_trend_df)
        mock_savefig.assert_called()  # Ensure savefig is called

        # Ensure that the plot file path is returned correctly
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        expected_file_path = os.path.join(plotter.output_subfolder, f'adjusted_feature_trends_stacked_bar_semiannual_{timestamp}.png')
        assert plot_file_path.startswith(expected_file_path.split('_')[0])  # Validate that timestamp is included

def test_invalid_data_in_plot_trends():
    # Testing with invalid data (non-numeric values)
    invalid_trend_df = pd.DataFrame({
        'year_month': pd.date_range(start='2020-01-01', periods=6, freq='ME'),
        'role1': [10, 20, 30, 'invalid', 50, 60],  # Invalid string data
        'role2': [5, 15, 25, 35, 45, 55],
        'role3': [0, 10, 20, 30, 40, 50]
    })

    plotter = SemiannualFeatureDistributionPlotter(
        role_columns=['role1', 'role2', 'role3'],
        output_subfolder='./plots',
        name_of_topics='Role'
    )

    with pytest.raises(Exception):  # Expecting an exception due to invalid data
        plotter.plot_trends(invalid_trend_df, pd.DataFrame())

def test_invalid_monthly_trend_data():
    # Testing with invalid monthly trend data (missing 'month' column)
    invalid_monthly_trend_df = pd.DataFrame({
        'role1_monthly': [5, 10, 15, 20, 25, 30],
        'role2_monthly': [2, 4, 6, 8, 10, 12],
        'role3_monthly': [0, 2, 4, 6, 8, 10]
    })

    plotter = SemiannualFeatureDistributionPlotter(
        role_columns=['role1', 'role2', 'role3'],
        output_subfolder='./plots',
        name_of_topics='Role'
    )

    with pytest.raises(KeyError):  # Expecting KeyError due to missing 'month' column
        plotter.plot_trends(pd.DataFrame(), invalid_monthly_trend_df)

def test_missing_columns_in_trend_df(plotter):
    # Test with missing expected columns in trend_df
    missing_columns_trend_df = pd.DataFrame({
        'year_month': pd.date_range(start='2020-01-01', periods=6, freq='ME'),
        'role1': [10, 20, 30, 40, 50, 60]
    })  # Missing 'role2' and 'role3'

    with pytest.raises(KeyError):  # Expecting KeyError due to missing columns
        plotter.plot_trends(missing_columns_trend_df, pd.DataFrame())

def test_missing_columns_in_monthly_trend_df(plotter):
    # Test with missing expected columns in monthly_trend_df
    missing_columns_monthly_trend_df = pd.DataFrame({
        'role1_monthly': [5, 10, 15, 20, 25, 30],
        'role2_monthly': [2, 4, 6, 8, 10, 12]
    })  # Missing 'month' and 'role3_monthly'

    with pytest.raises(KeyError):  # Expecting KeyError due to missing 'month' column
        plotter.plot_trends(pd.DataFrame(), missing_columns_monthly_trend_df)

def test_invalid_role_columns(plotter):
    # Test with invalid role columns that don't exist in trend_df
    invalid_role_columns = ['role4', 'role5', 'role6']  # These columns don't exist
    plotter_invalid = SemiannualFeatureDistributionPlotter(
        role_columns=invalid_role_columns,
        output_subfolder='./plots',
        name_of_topics='Role'
    )

    with pytest.raises(KeyError):  # Expecting KeyError due to missing role columns
        plotter_invalid.plot_trends(pd.DataFrame(), pd.DataFrame())

# Integration Test
def test_integration_plotter_with_data(plotter, trend_df, monthly_trend_df):
    # Simulate a complete workflow of plotting trends and ensuring all steps are covered
    with mock.patch('os.makedirs'), mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        plot_file_path = plotter.plot_trends(trend_df, monthly_trend_df)
        mock_savefig.assert_called()

        # Ensure that the plot file path is returned correctly
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        expected_file_path = os.path.join(plotter.output_subfolder, f'adjusted_feature_trends_stacked_bar_semiannual_{timestamp}.png')
        assert plot_file_path.startswith(expected_file_path.split('_')[0])  # Validate timestamp inclusion

def test_integration_plot_with_one_role_column():
    # Modify the plotter to use only one role column
    plotter_one_role = SemiannualFeatureDistributionPlotter(
        role_columns=['role1'],
        output_subfolder='./plots',
        name_of_topics='Role'
    )

    trend_df_one_role = pd.DataFrame({
        'year_month': pd.date_range(start='2020-01-01', periods=6, freq='ME'),
        'role1': [10, 20, 30, 40, 50, 60]
    })
    monthly_trend_df_one_role = pd.DataFrame({
        'month': [1, 2, 3, 4, 5, 6],
        'role1_monthly': [5, 10, 15, 20, 25, 30]
    })

    # Mocking os.makedirs and plt.savefig to avoid file creation
    with mock.patch('os.makedirs'), mock.patch('matplotlib.pyplot.savefig') as mock_savefig:
        plot_file_path = plotter_one_role.plot_trends(trend_df_one_role, monthly_trend_df_one_role)
        mock_savefig.assert_called()

        # Ensure the plot file path is returned correctly
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        expected_file_path = os.path.join(plotter_one_role.output_subfolder, f'adjusted_feature_trends_stacked_bar_semiannual_{timestamp}.png')
        assert plot_file_path.startswith(expected_file_path.split('_')[0])  # Validate timestamp inclusion