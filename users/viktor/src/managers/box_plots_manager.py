import pandas as pd
from datetime import datetime
import os
import json
import sys
from external_systems import SSEMEmbedder
from modules import KeywordFeatureExtractorBoxPlots, BoxPlotsVisualizer
from interfaces import IKeywordFeatureExtractor, IBoxPlots

from config import Config

# Load configuration
configs = Config()

keywords_folder_path = configs.keywords_folder_path
column_renames = configs.COLUMN_RENAMES
csv_dataset = configs.csv_dataset
text_column = configs.text_column
name_of_topics = configs.name_of_topics
reports_folder_path = configs.reports_folder_path

def get_json_files_for_box_plots():
    try:
        # Get all files in the specified folder
        files = [
            os.path.splitext(f)[0]  # Extract filename without extension
            for f in os.listdir(keywords_folder_path)
            if os.path.isfile(os.path.join(keywords_folder_path, f))
        ]

        if not files:
            print(f"No files found in the folder: {keywords_folder_path}")
            return []

        return files

    except FileNotFoundError:
        print(f"Error: Folder not found: {keywords_folder_path}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

class Box_Plots_Manager():
    def __init__(self, selected_folder, output_subfolder):
        self.selected_folder = selected_folder
        self.output_subfolder = output_subfolder
        self.topics_file = configs.topics_file

    def set_json_file_name(self, topics_file_name):
        topics_file = os.path.join(keywords_folder_path, topics_file_name)
        self.topics_file = topics_file

    def get_json_files_without_extension(self):
        try:
            # Get all files in the specified folder
            files = [
                os.path.splitext(f)[0]  # Extract filename without extension
                for f in os.listdir(keywords_folder_path)
                if os.path.isfile(os.path.join(keywords_folder_path, f))
            ]

            if not files:
                print(f"No files found in the folder: {keywords_folder_path}")
                return []

            return files

        except FileNotFoundError:
            print(f"Error: Folder not found: {keywords_folder_path}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def main(self):
        try:
            # Find a CSV file in the selected folder, excluding 'embeddings.csv'
            csv_files = [
                f for f in os.listdir(self.selected_folder)
                if f.endswith('.csv') and f != 'embeddings.csv'
            ]

            if not csv_files:
                raise FileNotFoundError("No valid CSV dataset files found in the selected folder.")

            # Use the first valid CSV file
            csv_dataset = os.path.join(self.selected_folder, csv_files[0])
            print(f"Using dataset: {csv_dataset}")

            # Find a CSV file in the selected folder called 'embeddings.csv'
            embedding_files = [
                f for f in os.listdir(self.selected_folder)
                if f.endswith('.csv') and f == 'embeddings.csv'
            ]

            if not embedding_files:
                raise FileNotFoundError("No valid CSV embeddings files found in the selected folder.")

            # Use the first valid CSV file
            embeddings_dataset = os.path.join(self.selected_folder, embedding_files[0])
            print(f"Using embeddings dataset: {embeddings_dataset}")

            # Load keyword dictionary from JSON
            json_file_path = self.topics_file

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
            # Load the dataset into a pandas DataFrame
            embeddings_df = pd.read_csv(embeddings_dataset)
            print(f"Dataset loaded with {len(embeddings_df)} records.")
            print(embeddings_df.columns)

        except FileNotFoundError:
            print(f"Error: Dataset file not found: {embeddings_dataset}")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: Dataset file is empty: {embeddings_dataset}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error loading dataset: {e}")
            sys.exit(1)

        df = df.rename(columns=column_renames)

        # Initialize the embedder
        embedder = SSEMEmbedder(model_name="all-mpnet-base-v2")

        # Generate embeddings for the keywords
        keyword_embeddings = {
            feature_name: embedder.generate_embeddings(keywords)
            for feature_name, keywords in keyword_dict.items()
        }

        text_column = 'description_embeddings'

        try:
            # Initialize keyword feature extractor using the interface IKeywordFeatureExtractor
            kfe: IKeywordFeatureExtractor = KeywordFeatureExtractorBoxPlots(df, embeddings_df, text_column, keyword_dict, keyword_embeddings, temp=0.5)

            # Apply keyword feature extraction
            df_features = kfe.extract_features()

            print(df_features.head())

            role_columns = list(keyword_dict.keys()) + ['Other']
            df_features['original_listed_time'] = pd.to_datetime(df_features['original_listed_time'], errors='coerce')
            if df_features['original_listed_time'].isnull().all():
                raise ValueError("All 'original_listed_time' values are invalid or missing.")

            df_features['year_month'] = df_features['original_listed_time'].dt.to_period('M')

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
            box_plots_visualizer: IBoxPlots = BoxPlotsVisualizer(trend_df, monthly_trend_df, role_columns, self.output_subfolder, reports_folder_path, name_of_topics)

            # Plot the feature percentage distribution
            box_plots_visualizer.plot_distribution()

        except Exception as e:
            print(f"Unexpected error during visualization: {e}")
            sys.exit(1)

        print("Process completed successfully.")