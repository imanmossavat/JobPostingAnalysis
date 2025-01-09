import os
import pandas as pd
import json

from config import Config

from external_systems import SSEMEmbedder
from modules import WordCloudGenerator
from interfaces import IWordCloudGenerator

from datetime import datetime

# Load configuration
configs = Config()

keywords_folder_path = configs.keywords_folder_path
text_column = configs.text_column
stopword_file_names = configs.stopword_file_names
# csv_dataset = configs.csv_dataset
name_of_topics = configs.name_of_topics
reports_folder_path = configs.reports_folder_path

def get_json_files_for_word_clouds():
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

class Word_Clouds_Manager():
    """
    Manages the generation of WordClouds for topics defined in a JSON file, based on a CSV dataset.

    Args:
        output_folder (str): Name of the output folder to save the generated WordClouds.

    Raises:
        FileNotFoundError: If the CSV dataset or topics JSON file cannot be found.
        ValueError: If the CSV dataset is empty or the JSON file is invalid.
        RuntimeError: If there is an issue creating directories, loading files, or generating WordClouds.
    """
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

        try:
            # Load the CSV dataset
            data = pd.read_csv(csv_dataset)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV dataset not found at path: {csv_dataset}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The CSV dataset at {csv_dataset} is empty.")
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV dataset: {e}")
        
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
        
        try:
            # Load the CSV dataset
            embeddings_data = pd.read_csv(embeddings_dataset)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV embeddings dataset not found at path: {embeddings_dataset}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The CSV embeddings dataset at {embeddings_dataset} is empty.")
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV embeddings dataset: {e}")
        
        # Ensure the output folder is inside the 'reports' directory
        try:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            output_subfolder_path = os.path.join(reports_folder_path, f'{self.output_subfolder}_{timestamp}')
            os.makedirs(output_subfolder_path, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory: {e}")

        # Load topics and keywords from the JSON file
        try:
            with open(self.topics_file, 'r') as file:
                keyword_dict = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Topics JSON file not found at path: {self.topics_file}")
        except json.JSONDecodeError:
            raise ValueError(f"The topics JSON file at {self.topics_file} is not a valid JSON format.")
        except Exception as e:
            raise RuntimeError(f"Failed to load topics JSON file: {e}")

        # Ensure keyword_dict is valid
        if not isinstance(keyword_dict, dict) or not keyword_dict:
            raise ValueError("The topics JSON file does not contain a valid dictionary of topics and keywords.")

        # Initialize the embedder
        embedder = SSEMEmbedder(model_name="all-mpnet-base-v2")

        # Generate embeddings for the keywords
        keyword_embeddings = {
            feature_name: embedder.generate_embeddings(keywords)
            for feature_name, keywords in keyword_dict.items()
        }

        # text_column = 'description_embeddings'

        # Initialize the WordCloudGenerator
        try:
            generator :IWordCloudGenerator = WordCloudGenerator(data, embeddings_data, keyword_dict, output_subfolder_path, name_of_topics, stopword_file_names, text_column)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WordCloudGenerator: {e}")

        # Generate the WordCloud images
        try:
            image_paths = generator.generate_wordcloud_for_topic()
        except Exception as e:
            raise RuntimeError(f"Error generating WordCloud images: {e}")

        # Ensure images were generated successfully
        if not image_paths:
            raise RuntimeError("No WordCloud images were generated. Please check the input data and topics.")

        # Log the paths of the generated images
        try:
            print("WordCloud images saved at the following paths:")
            for path in image_paths:
                print(path)
        except Exception as e:
            raise RuntimeError(f"Error logging generated image paths: {e}")