import os
import sys
import pandas as pd
import numpy as np
import json
import re
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
from datetime import datetime
from interfaces import IKeywordFeatureExtractor, IBoxPlots
from sklearn.metrics.pairwise import cosine_similarity

class KeywordFeatureExtractorBoxPlots(IKeywordFeatureExtractor):
    """
    A class for extracting keyword-based features from a DataFrame containing embeddings and applying softmax normalization.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing the embeddings.
        column (str): The name of the column in the DataFrame that contains embeddings (e.g., sentence embeddings).
        keyword_dict (dict): A dictionary where keys are feature names and values are lists of keywords.
        keyword_embeddings (dict): A dictionary where the keys are feature names and the values are the embeddings for the keywords.
        temp (float): Temperature parameter for softmax normalization, which controls the scale of the values (default is 0.5).

    Methods:
        extract_features():
            Extracts features based on cosine similarity between the embeddings and keyword embeddings, then applies softmax normalization.
    """
    
    def __init__(self, df, embeddings_df, column, keyword_dict, keyword_embeddings, temp=0.5):
        """
        Initializes the KeywordFeatureExtractorBoxPlots instance.

        Args:
            df (pd.DataFrame): DataFrame containing embeddings.
            column (str): Name of the column containing embeddings.
            keyword_dict (dict): Dictionary with feature names as keys and lists of keywords as values.
            keyword_embeddings (dict): Dictionary with feature names as keys and their corresponding keyword embeddings as values.
            temp (float): Temperature parameter for softmax normalization (default is 0.5).
        """
        self.df = df
        self.embeddings_df = embeddings_df
        self.column = column
        self.keyword_dict = keyword_dict
        self.keyword_embeddings = keyword_embeddings
        self.temp = temp
    
    def extract_features(self):
        """
        Extracts features based on cosine similarity between embeddings and keyword embeddings, and normalizes the features using softmax.
        """
        # Check if column exists
        if self.column not in self.embeddings_df.columns:
            print(f"Column '{self.column}' not found in DataFrame.")
            sys.exit(1)

        # Ensure the embeddings are in the correct format (convert string representation to list)
        embeddings = np.array(self.embeddings_df[self.column].apply(lambda x: np.array(ast.literal_eval(x))).tolist())

        # Create a copy of the original DataFrame to store the features
        feature_df = self.df.copy()

        # Calculate the cosine similarity between each embedding and the keyword embeddings
        for feature_name, keywords in self.keyword_dict.items():
            # Get the keyword embeddings
            keyword_embeds = self.keyword_embeddings[feature_name]

            # Calculate the cosine similarity for each embedding in the column
            similarities = cosine_similarity(embeddings, keyword_embeds)

            # Assign the maximum similarity to the corresponding feature
            feature_df[feature_name] = np.max(similarities, axis=1)

        # Add 'Other' feature for no match (this will be handled by the lowest similarity values)
        feature_df['Other'] = 1 - feature_df[list(self.keyword_dict.keys())].max(axis=1)

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
        semiannual_trends_df = adjusted_trend_df[self.role_columns].resample('6ME').sum()
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
        sns.boxplot(
            data=melted_data,
            x=self.name_of_topics,
            y='Percentage',
            order=feature_order_median,
            palette='tab20',
            hue=self.name_of_topics,
            dodge=False,  # Avoid splitting the bars by hue
            legend=False  # Suppress legend as it is redundant here
        )
        plt.title(f'{self.name_of_topics} Percentages Distribution (Ordered by Median)', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subfolder, 'feature_percentage_distribution_boxplot_median.png'))
        plt.close()

        # Create and save the box plot ordered by mean
        plt.figure(figsize=(14, 8))
        sns.boxplot(
            data=melted_data,
            x=self.name_of_topics,
            y='Percentage',
            order=feature_order_mean,
            palette='tab20',
            hue=self.name_of_topics,
            dodge=False,  # Avoid splitting the bars by hue
            legend=False  # Suppress legend as it is redundant here
        )
        plt.title(f'{self.name_of_topics} Percentages Distribution (Ordered by Mean)', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subfolder, 'feature_percentage_distribution_boxplot_mean.png'))
        plt.close()