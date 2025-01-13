import os
import pandas as pd
from datetime import datetime

from config import Config
from modules import TopicModel, TopicModelVisualizer
from interfaces import ITopicModel, ITopicModelVisualizer

# Load the dataset path and stopword files from the configuration
configs = Config()

text_column = configs.text_column

stopword_file_names = configs.stopword_file_names
reports_folder_path = configs.reports_folder_path

class Topic_Modeling_Manager():
    def __init__(self, selected_folder, output_folder, n_topics, num_top_words, epochs):
        self.selected_folder = selected_folder
        self.output_folder = output_folder
        self.n_topics = n_topics
        self.num_top_words = num_top_words
        self.epochs = epochs
        self.column_name = text_column

    def main(self):
        # Find a CSV file in the selected folder
        normal_files = [
            f for f in os.listdir(self.selected_folder)
            if f.endswith('.csv') and f != 'embeddings.csv'
        ]

        if not normal_files:
            raise FileNotFoundError("No valid CSV dataset files found in the selected folder.")
        
        # Use the first valid CSV file
        normal_dataset = os.path.join(self.selected_folder, normal_files[0])
        print(f"Using dataset: {normal_dataset}")

        try:
            # Load the CSV dataset
            normal_data = pd.read_csv(normal_dataset)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV dataset not found at path: {normal_dataset}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The CSV dataset at {normal_dataset} is empty.")
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV dataset: {e}")

        # Find a CSV file in the selected folder which is'embeddings.csv'
        embedding_files = [
            f for f in os.listdir(self.selected_folder)
            if f.endswith('.csv') and f == 'embeddings.csv'
        ]

        if not embedding_files:
            raise FileNotFoundError("No valid CSV dataset files found in the selected folder.")
        
        # Use the first valid CSV file
        embeddings_dataset = os.path.join(self.selected_folder, embedding_files[0])
        print(f"Using dataset: {embeddings_dataset}")

        try:
            # Load the CSV dataset
            embeddings_data = pd.read_csv(embeddings_dataset)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV dataset not found at path: {embeddings_dataset}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The CSV dataset at {embeddings_dataset} is empty.")
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV dataset: {e}")
   
        try:
            # Get current timestamp for unique folder naming
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            
            # Ensure the output folder is inside the 'reports' directory and include the timestamp
            output_folder_path = os.path.join(reports_folder_path, f'{self.output_folder}_{timestamp}')
            
            # Create the 'reports' folder if it doesn't already exist
            os.makedirs(output_folder_path, exist_ok=True)

            texts = normal_data[self.column_name]
            
            self.column_name = 'description_embeddings'

            model = "all-mpnet-base-v2"

            embeddings = embeddings_data[self.column_name]

            topic_model: ITopicModel = TopicModel(
                embeddings=embeddings,
                texts=texts,
                n_topics=self.n_topics,
                num_keywords=self.num_top_words,
                max_iter=self.epochs,
                model=model,
                output_subfolder=output_folder_path,
            )

            topic_model.fit_model()
            
            # Execute topic modeling and display the topics
            metrics_data, keywords, desc_embeddings = topic_model.execute_topic_modeling()

        except ValueError as ve:
            # Handle errors during the NMF model initialization or execution
            print(f"Error with topic model initialization or execution: {ve}")
            return None
        except FileNotFoundError as fnf_error:
            # Handle missing file errors, like missing stopword files
            print(f"Error: {fnf_error}")
            return None
        except Exception as e:
            # Handle any other unexpected errors during the model fitting or topic displaying
            print(f"An unexpected error occurred: {e}")
            return None

        experiment_file = "experiment_file.txt"

        # Visualize the topics and evaluation metrics
        visualizer: ITopicModelVisualizer = TopicModelVisualizer(topic_model, output_folder_path, experiment_file)
        visualizer.create_all_visualizations(
            metrics_data["topic_diversity_score"], 
            metrics_data["silhouette_score"], 
            metrics_data["clustering_stability"], 
            metrics_data["topic_percentages"],
            metrics_data["inertia"],
            metrics_data["adjusted_rand_index"],
            keywords
        )