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
    """
    A class for extracting keyword-based features from a text column in a DataFrame and applying softmax normalization.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing the text data.
        column (str): The name of the column in the DataFrame that contains text (e.g., job descriptions).
        keyword_dict (dict): A dictionary where keys are feature names and values are lists of keywords.
        temp (float): Temperature parameter for softmax normalization, which controls the scale of the values (default is 0.5).

    Methods:
        extract_features():
            Extracts binary features based on keyword matches in the specified text column and applies softmax normalization.
    """
    
    def __init__(self, df, column, keyword_dict, temp=0.5):
        """
        Initializes the KeywordFeatureExtractorBoxPlots instance.

        Args:
            df (pd.DataFrame): DataFrame containing text data.
            column (str): Name of the text column to process.
            keyword_dict (dict): Dictionary with feature names as keys and lists of keywords as values.
            temp (float): Temperature parameter for softmax normalization (default is 0.5).
        """
        self.df = df
        self.column = column
        self.keyword_dict = keyword_dict
        self.temp = temp

    def extract_features(self):
        """
        Extracts binary features based on keyword matches from the specified text column and normalizes the features using softmax.

        Each feature corresponds to a keyword group, and a value of 1 is assigned if any of the keywords in the group are 
        found in the text. An 'Other' feature is also created for entries that do not match any of the keywords.

        The features are then normalized using the softmax function, where the 'temp' parameter is used for scaling the values.

        Returns:
            pd.DataFrame: A DataFrame with the original data and the newly extracted features, with softmax normalization applied.
        """
        # Check if column exists
        if self.column not in self.df.columns:
            print(f"Column '{self.column}' not found in DataFrame.")
            sys.exit(1)

        # Fill missing values with empty strings
        if self.df[self.column].isnull().any():
            self.df[self.column].fillna('', inplace=True)

        # Create a copy of the original DataFrame to store the features
        feature_df = self.df.copy()

        # Extract binary features for each keyword group
        for feature_name, keywords in self.keyword_dict.items():
            feature_df[feature_name] = feature_df[self.column].apply(
                lambda text: int(any(re.search(r'\b' + re.escape(keyword) + r'\b', text, flags=re.IGNORECASE) for keyword in keywords))
            )

        # Add 'Other' feature for no keyword match
        feature_df['Other'] = (feature_df[list(self.keyword_dict.keys())].sum(axis=1) == 0).astype(int)

        # Normalize the features using softmax
        def apply_softmax_with_temp(row):
            return softmax(np.array(row) / self.temp)

        # Apply softmax to the features
        feature_df[list(self.keyword_dict.keys()) + ['Other']] = feature_df[list(self.keyword_dict.keys()) + ['Other']].apply(
            apply_softmax_with_temp, axis=1, result_type='expand')

        return feature_df

class BoxPlotsVisualizer(IBoxPlots):
    """
    A class for visualizing the distribution of feature percentages over time using box plots.

    Attributes:
        trend_df (pd.DataFrame): DataFrame containing trend data over time.
        monthly_trend_df (pd.DataFrame): DataFrame with monthly trend data.
        role_columns (list): A list of column names that represent different features (e.g., job roles).
        output_subfolder_base (str): Base name for the output subfolder where reports will be saved.
        reports_folder_path (str): Path to the folder where reports will be stored.

    Methods:
        plot_distribution():
            Creates and saves box plots visualizing the distribution of feature percentages, with the option to order by median or mean.
    """
    
    def __init__(self, trend_df, monthly_trend_df, role_columns, output_subfolder_base, reports_folder_path, name_of_topics):
        """
        Initializes the BoxPlotsVisualizer instance.

        Args:
            trend_df (pd.DataFrame): DataFrame containing trend data over time.
            monthly_trend_df (pd.DataFrame): DataFrame with monthly trend data.
            role_columns (list): List of column names representing different roles or features.
            output_subfolder_base (str): Base name for the output subfolder for saving plots.
            reports_folder_path (str): Path to the folder where the reports will be saved.
        """
        self.trend_df = trend_df
        self.monthly_trend_df = monthly_trend_df
        self.role_columns = role_columns
        self.output_subfolder_base = output_subfolder_base
        self.reports_folder_path = reports_folder_path
        self.name_of_topics = name_of_topics

    def plot_distribution(self):
        """
        Creates box plots to visualize the distribution of feature percentages based on semi-annual trends.

        The box plots represent the percentage distribution of each feature (role) over time, ordered either by the median
        or the mean value of the feature percentages. The results are saved as PNG images in a timestamped subfolder.

        The trends are resampled at 6-month intervals to give a semi-annual view.

        Returns:
            None: This method generates and saves the box plots, it does not return any value.
        """
        # Define the parent folder and create a timestamped subfolder for saving the plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subfolder = os.path.join(self.reports_folder_path, f"{self.output_subfolder_base}_{timestamp}")
        
        # Ensure the output directory exists
        os.makedirs(output_subfolder, exist_ok=True)

        # Extract the month from 'year_month' and merge with monthly trends
        self.trend_df['month'] = self.trend_df['year_month'].dt.month
        adjusted_trend_df = self.trend_df.merge(self.monthly_trend_df, on='month', suffixes=('', '_monthly'))
        adjusted_trend_df.set_index('year_month', inplace=True)

        # Resample the data by 6-month intervals and calculate percentage distribution
        semiannual_trends_df = adjusted_trend_df[self.role_columns].resample('6M').sum()
        semiannual_trends_percent_df = semiannual_trends_df.div(semiannual_trends_df.sum(axis=1), axis=0) * 100

        # Melt the data for plotting
        melted_data = semiannual_trends_percent_df.reset_index().melt(
            id_vars='year_month',
            value_vars=self.role_columns,
            var_name=self.name_of_topics,
            value_name='Percentage'
        )

        # Order features by median and mean of the percentage
        feature_order_median = melted_data.groupby(self.name_of_topics)['Percentage'].median().sort_values(ascending=False).index
        feature_order_mean = melted_data.groupby(self.name_of_topics)['Percentage'].mean().sort_values(ascending=False).index

        # Create and save the box plot ordered by median
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=melted_data, x=self.name_of_topics, y='Percentage', order=feature_order_median, palette='tab20')
        plt.title(f'{self.name_of_topics} Percentages Distribution (Ordered by Median)', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subfolder, 'feature_percentage_distribution_boxplot_median.png'))
        plt.close()

        # Create and save the box plot ordered by mean
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=melted_data, x=self.name_of_topics, y='Percentage', order=feature_order_mean, palette='tab20')
        plt.title(f'{self.name_of_topics} Percentages Distribution (Ordered by Mean)', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subfolder, 'feature_percentage_distribution_boxplot_mean.png'))
        plt.close()