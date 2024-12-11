import os
import sys
import pandas as pd
import numpy as np
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
from datetime import datetime
from interfaces import IKeywordFeatureExtractor, IBoxPlots

class KeywordFeatureExtractorBoxPlots(IKeywordFeatureExtractor):
    def __init__(self):
        # Initialize any necessary attributes if required
        pass

    def extract_features(self, df, column, keyword_dict, temp=0.5):
        """
        Extracts features from a text column in a DataFrame based on a keyword dictionary.
        Applies softmax to normalize feature values.

        Args:
            df (pd.DataFrame): DataFrame containing text data.
            column (str): Column name with the job descriptions.
            keyword_dict (dict): A dictionary with feature names as keys and lists of keywords as values.
            temp (float): Temperature for softmax scaling, default is 0.5.

        Returns:
            pd.DataFrame: Updated DataFrame with new features extracted from keywords.
        """
        # Check if column exists
        if column not in df.columns:
            print(f"Column '{column}' not found in DataFrame.")
            sys.exit(1)

        if df[column].isnull().any():
            df[column].fillna('', inplace=True)

        # Create a copy to store features
        feature_df = df.copy()

        for feature_name, keywords in keyword_dict.items():
            # Add a binary feature for each keyword group
            feature_df[feature_name] = feature_df[column].apply(
                lambda text: int(any(re.search(r'\b' + re.escape(keyword) + r'\b', text, flags=re.IGNORECASE) for keyword in keywords))
            )

        # Add 'Other' column for no keywords match
        feature_df['Other'] = (feature_df[list(keyword_dict.keys())].sum(axis=1) == 0).astype(int)

        # Apply softmax normalization
        def apply_softmax_with_temp(row):
            return softmax(np.array(row) / temp)

        feature_df[list(keyword_dict.keys()) + ['Other']] = feature_df[list(keyword_dict.keys()) + ['Other']].apply(
            apply_softmax_with_temp, axis=1, result_type='expand')

        return feature_df

class BoxPlotsVisualizer(IBoxPlots):
    def __init__(self):
        # Initialize any necessary attributes if required
        pass

    def plot_distribution(self, trend_df, monthly_trend_df, role_columns, output_subfolder_base, reports_folder_path):
        """
        Creates box plots of feature percentages distribution based on semi-annual trends.

        Args:
            trend_df (pd.DataFrame): DataFrame with trends.
            monthly_trend_df (pd.DataFrame): DataFrame with monthly trends.
            role_columns (list): List of role columns to plot.
            output_subfolder_base (str): Base name for the output subfolder.

        Returns:
            None: Generates and saves the box plots.
        """
        # Define the parent folder and generate a timestamped folder name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subfolder = os.path.join(reports_folder_path, f"{output_subfolder_base}_{timestamp}")
        
        # Ensure the output directory exists
        os.makedirs(output_subfolder, exist_ok=True)

        trend_df['month'] = trend_df['year_month'].dt.month
        adjusted_trend_df = trend_df.merge(monthly_trend_df, on='month', suffixes=('', '_monthly'))
        adjusted_trend_df.set_index('year_month', inplace=True)

        # Resample by 6-month intervals
        semiannual_trends_df = adjusted_trend_df[role_columns].resample('6M').sum()
        semiannual_trends_percent_df = semiannual_trends_df.div(semiannual_trends_df.sum(axis=1), axis=0) * 100

        melted_data = semiannual_trends_percent_df.reset_index().melt(
            id_vars='year_month',
            value_vars=role_columns,
            var_name='Feature',
            value_name='Percentage'
        )

        feature_order_median = melted_data.groupby('Feature')['Percentage'].median().sort_values(ascending=False).index
        feature_order_mean = melted_data.groupby('Feature')['Percentage'].mean().sort_values(ascending=False).index

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=melted_data, x='Feature', y='Percentage', order=feature_order_median, palette='tab20')
        plt.title('Feature Percentages Distribution (Ordered by Median)', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subfolder, 'feature_percentage_distribution_boxplot_median.png'))
        plt.close()

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=melted_data, x='Feature', y='Percentage', order=feature_order_mean, palette='tab20')
        plt.title('Feature Percentages Distribution (Ordered by Mean)', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subfolder, 'feature_percentage_distribution_boxplot_mean.png'))
        plt.close()