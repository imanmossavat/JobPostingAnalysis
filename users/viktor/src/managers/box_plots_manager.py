import pandas as pd
from datetime import datetime
import os
import json
import sys
from modules import KeywordFeatureExtractorBoxPlots, BoxPlotsVisualizer
from interfaces import IKeywordFeatureExtractor, IBoxPlots

from config import Config

# Load configuration
configs = Config()

topics_file = configs.topics_file
csv_dataset = configs.csv_dataset
text_column = configs.text_column
name_of_topics = configs.name_of_topics
reports_folder_path = configs.reports_folder_path

def Box_Plots_Manager(output_subfolder):
    try:
        # Load keyword dictionary from JSON
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, topics_file)

        if not os.path.isfile(json_file_path):
            print(f"Error: File does not exist: {json_file_path}")
            sys.exit(1)

        with open(json_file_path, 'r') as file:
            keyword_dict = json.load(file)

    except json.JSONDecodeError as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading JSON file: {e}")
        sys.exit(1)

    try:
        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(csv_dataset)
        print(f"Dataset loaded with {len(df)} records.")
        print(df.columns)

    except FileNotFoundError:
        print(f"Error: Dataset file not found: {csv_dataset}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Dataset file is empty: {csv_dataset}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        sys.exit(1)

    try:
        # Initialize keyword feature extractor using the interface IKeywordFeatureExtractor
        kfe: IKeywordFeatureExtractor = KeywordFeatureExtractorBoxPlots(df, text_column, keyword_dict, temp=0.5)

        # Apply keyword feature extraction
        df_features = kfe.extract_features()

        print(df_features.head())

        role_columns = list(keyword_dict.keys()) + ['Other']
        df_features['CreatedAt'] = pd.to_datetime(df_features['CreatedAt'], errors='coerce')
        if df_features['CreatedAt'].isnull().all():
            raise ValueError("All 'CreatedAt' values are invalid or missing.")

        df_features['year_month'] = df_features['CreatedAt'].dt.to_period('M')

        # Group by year and month for trend analysis
        trend_df = df_features.groupby('year_month')[role_columns].sum().reset_index()
        trend_df['year_month'] = trend_df['year_month'].dt.to_timestamp()

        # Monthly trend analysis
        df_features['month'] = df_features['year_month'].dt.month
        monthly_trend_df = df_features.groupby('month')[role_columns].mean().reset_index()
        monthly_trend_df['month'] = monthly_trend_df['month'].astype('category')

    except KeyError as e:
        print(f"KeyError: Missing column in dataset - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ValueError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during data processing: {e}")
        sys.exit(1)

    try:
        # Initialize visualizer using the interface IBoxPlots
        box_plots_visualizer: IBoxPlots = BoxPlotsVisualizer(trend_df, monthly_trend_df, role_columns, output_subfolder, reports_folder_path, name_of_topics)

        # Plot the feature percentage distribution
        box_plots_visualizer.plot_distribution()

    except Exception as e:
        print(f"Unexpected error during visualization: {e}")
        sys.exit(1)

    print("Process completed successfully.")