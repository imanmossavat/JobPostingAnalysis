from modules import DatasetRegistry, DataFormatter 
from interfaces import IDataFormatter
from config import Config
from external_systems import SSEMEmbedder

import pandas as pd
import json
import numpy as np

# Load configuration
configs = Config()

base_folder = configs.base_folder
registry_file = configs.registry_file

column_renames = configs.COLUMN_RENAMES
special_handlings_columns = configs.SPECIAL_HANDLINGS_COLUMNS

class DataRegistryManager:
    """
    Wrapper for DatasetRegistry to manage datasets and database connections.
    """
    
    def __init__(self, dataset, file_name, project_name):
        self.project_name = project_name
        self.dataset = dataset
        self.file_name = file_name
        self.dataset_registry = DatasetRegistry(dataset, project_name, file_name, base_folder, registry_file)
    
    def save_dataset(self, dataset, project_name, db_connection=None):
        """
        Save a dataset or database connection.
        """
        if db_connection:
            # Save database connection details
            dataset_name = f"{db_connection['name']}_db"
            return self.dataset_registry.save_dataset(None, dataset_name, project_name, db_connection)
        
        if dataset is not None:
            if dataset.name.endswith('.csv'):
                df = pd.read_csv(dataset)
            elif dataset.name.endswith('.xlsx'):
                df = pd.read_excel(dataset)
            elif dataset.name.endswith('.json'):
                df = pd.read_json(dataset)
            else:
                return "Unsupported file format."

            data_formatter: IDataFormatter = DataFormatter(df, column_renames, special_handlings_columns)
            df = data_formatter.rename_columns()

            # Initialize embedder
            embedder = SSEMEmbedder(model_name="all-mpnet-base-v2")

            # Generate embeddings for the 'description' column
            descriptions = df["description"].tolist()
            embeddings = embedder.generate_embeddings(descriptions)

            ''' # Save embeddings to a file
            np.save("description_embeddings2.npy", embeddings) '''

            embeddings = np.array(embeddings)  # Ensure embeddings are in NumPy array format

            # Create a DataFrame for embeddings
            embeddings_df = pd.DataFrame(embeddings)

            # Save to a CSV file
            embeddings_df.to_csv("description_embeddings.csv", index=False, header=False)

            dataset_name = dataset.name
            return self.dataset_registry.save_dataset(df, dataset_name, project_name)

    def remove_dataset(self, project_to_remove, dataset_to_remove):
        return self.dataset_registry.remove_dataset(project_to_remove, dataset_to_remove)

    def get_existing_projects(self):
        return self.dataset_registry.get_existing_projects()

    def get_datasets_in_project(self, project_to_remove):
        return self.dataset_registry.get_datasets_in_project(project_to_remove)