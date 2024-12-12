import os
import pandas as pd
import json

from config import Config

from modules.word_clouds import WordCloudGenerator
from datetime import datetime

# Load configuration
configs = Config()

stopword_file_names = configs.stopword_file_names
csv_dataset = configs.csv_dataset
name_of_topics = configs.name_of_topics
topics_file = configs.topics_file
reports_folder_path = configs.reports_folder_path

def Word_Clouds_Manager(output_folder):
    """
    Manages the generation of WordClouds for topics defined in a JSON file, based on a CSV dataset.

    Args:
        output_folder (str): Name of the output folder to save the generated WordClouds.

    Raises:
        FileNotFoundError: If the CSV dataset or topics JSON file cannot be found.
        ValueError: If the CSV dataset is empty or the JSON file is invalid.
        RuntimeError: If there is an issue creating directories, loading files, or generating WordClouds.
    """
    try:
        # Load the CSV dataset
        data = pd.read_csv(csv_dataset)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV dataset not found at path: {csv_dataset}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The CSV dataset at {csv_dataset} is empty.")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV dataset: {e}")

    # Ensure the output folder is inside the 'reports' directory
    try:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_folder_path = os.path.join(reports_folder_path, f'{output_folder}_{timestamp}')
        os.makedirs(output_folder_path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory: {e}")

    # Load topics and keywords from the JSON file
    try:
        with open(topics_file, 'r') as file:
            keyword_dict = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Topics JSON file not found at path: {topics_file}")
    except json.JSONDecodeError:
        raise ValueError(f"The topics JSON file at {topics_file} is not a valid JSON format.")
    except Exception as e:
        raise RuntimeError(f"Failed to load topics JSON file: {e}")

    # Ensure keyword_dict is valid
    if not isinstance(keyword_dict, dict) or not keyword_dict:
        raise ValueError("The topics JSON file does not contain a valid dictionary of topics and keywords.")

    # Initialize the WordCloudGenerator
    try:
        generator = WordCloudGenerator()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize WordCloudGenerator: {e}")

    # Generate the WordCloud images
    try:
        image_paths = generator.generate_wordcloud_for_topic(
            data, keyword_dict, output_folder_path, name_of_topics, stopword_file_names
        )
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