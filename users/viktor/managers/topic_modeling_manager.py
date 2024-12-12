import os
from config import Config
from modules import NMFModel
import pandas as pd
from datetime import datetime

# Load the dataset path and stopword files from the configuration
configs = Config()

csv_dataset = configs.csv_dataset
stopword_file_names = configs.stopword_file_names
reports_folder_path = configs.reports_folder_path

def Topic_Modeling_Manager(output_folder, n_topics, num_top_words, epochs):
    """
    Manages the topic modeling process by loading the dataset, initializing the NMF model,
    and displaying the discovered topics.

    Args:
        output_folder (str): The folder name under the 'reports' directory where the output will be saved.
        n_topics (int): The number of topics to extract.
        num_top_words (int): The number of top words to display for each topic.
        epochs (int): The number of training epochs for the NMF model.

    Returns:
        list: A list of topics (words per topic) if the process is successful, otherwise None.
    """
    try:
        # Load the dataset
        if not os.path.exists(csv_dataset):
            raise FileNotFoundError(f"The dataset file '{csv_dataset}' was not found.")
        
        # Read the dataset into a pandas DataFrame
        data = pd.read_csv(csv_dataset)
        
        if data.empty:
            raise ValueError("The dataset is empty. Please provide a valid dataset.")
        
    except FileNotFoundError as fnf_error:
        # Handle file not found error
        print(f"Error: {fnf_error}")
        return None
    except ValueError as ve:
        # Handle empty dataset error
        print(f"Error: {ve}")
        return None
    except Exception as e:
        # Handle any other unexpected errors during dataset loading
        print(f"An unexpected error occurred while loading the dataset: {e}")
        return None

    try:
        # Get current timestamp for unique folder naming
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # Ensure the output folder is inside the 'reports' directory and include the timestamp
        output_folder_path = os.path.join(reports_folder_path, f'{output_folder}_{timestamp}')
        
        # Create the 'reports' folder if it doesn't already exist
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Initialize the NMF model with the specified parameters
        nmf_model = NMFModel(
            n_topics=n_topics, 
            data=data, 
            stopword_files=stopword_file_names, 
            num_top_words=num_top_words, 
            epochs=epochs, 
            output_subfolder=output_folder_path
        )
        
        # Fit the NMF model to the data and display the topics
        nmf_model.fit()
        topics = nmf_model.display_topics()
        
        return topics

    except ValueError as ve:
        # Handle errors during the NMF model initialization or execution
        print(f"Error with NMF model initialization or execution: {ve}")
        return None
    except FileNotFoundError as fnf_error:
        # Handle missing file errors, like missing stopword files
        print(f"Error: {fnf_error}")
        return None
    except Exception as e:
        # Handle any other unexpected errors during the model fitting or topic displaying
        print(f"An unexpected error occurred: {e}")
        return None