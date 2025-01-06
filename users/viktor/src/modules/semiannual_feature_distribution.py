import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from interfaces import ISemiannualFeatureDistribution

class SemiannualFeatureDistributionPlotter(ISemiannualFeatureDistribution):
    """
    Implementation of ISemiannualFeatureDistribution for semiannual feature distribution plots.

    Attributes:
        role_columns (list): List of feature column names to include in the plot.
        output_subfolder (str): Directory where the plot image will be saved.
        name_of_topics (str): Descriptive name of the features/topics being plotted.
    """

    def __init__(self, role_columns: list, output_subfolder: str, name_of_topics: str):
        """
        Initialize the SemiannualFeatureDistributionPlotter.

        Args:
            role_columns (list): List of feature column names.
            output_subfolder (str): Directory to save the plot.
            name_of_topics (str): Name of the topics for the plot title.
        """
        self.role_columns = role_columns
        self.output_subfolder = output_subfolder
        self.name_of_topics = name_of_topics

    def plot_trends(self, trend_df: pd.DataFrame, monthly_trend_df: pd.DataFrame) -> str:
        """
        Generate a semiannual feature distribution plot.

        Args:
            trend_df (pd.DataFrame): DataFrame containing trend data by year and month.
            monthly_trend_df (pd.DataFrame): DataFrame containing monthly adjustments.

        Returns:
            str: File path to the saved plot image.
        """

        # Ensure the output directory exists
        # os.makedirs(self.output_subfolder, exist_ok=True)

        trend_df['month'] = trend_df['year_month'].dt.month
        adjusted_trend_df = trend_df.merge(monthly_trend_df, on='month', suffixes=('', '_monthly'))

        adjusted_trend_df['year_month'] = pd.to_datetime(adjusted_trend_df['year_month'])
        adjusted_trend_df.set_index('year_month', inplace=True)

        semiannual_trends_df = adjusted_trend_df[self.role_columns].resample('6ME').sum()
        semiannual_trends_percent_df = semiannual_trends_df.div(semiannual_trends_df.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(22, 14))
        semiannual_trends_percent_df.plot(kind='bar', stacked=True, width=0.9, colormap='tab20', ax=ax)

        ax.set_title(f'Percentage Distribution of {self.name_of_topics} Adjusted Feature Trends Over Time', fontsize=24, fontweight='bold', pad=20)
        ax.set_xlabel('Year-Half', fontsize=20, labelpad=15)
        ax.set_ylabel('Percentage (%)', fontsize=20, labelpad=15)

        x_labels = [f'{date.year} H{1 if date.month <= 6 else 2}' for date in semiannual_trends_percent_df.index]
        ax.set_xticks(np.arange(len(semiannual_trends_percent_df.index)))
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=16)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_yticklabels([f"{int(tick)}%" for tick in ax.get_yticks()], fontsize=16)

        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', linewidth=0.7)
        ax.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, title_fontsize=18)

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                width = p.get_width()
                x_position = p.get_x() + width / 2
                y_position = p.get_y() + height / 2
                percentage = f'{height:.1f}%'
                ax.annotate(percentage, (x_position, y_position), ha='center', va='center', fontsize=14, color='black')

        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file_path = os.path.join(self.output_subfolder, f'adjusted_feature_trends_stacked_bar_semiannual_{timestamp}.png')
        plt.savefig(plot_file_path)
        return plot_file_path