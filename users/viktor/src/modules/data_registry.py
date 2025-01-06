import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from interfaces import IDatasetRegistry

class DatasetRegistry(IDatasetRegistry):
    """
    A class to manage datasets and database connections in a structured way.
    """

    def __init__(self, dataset, project_name, dataset_name, BASE_FOLDER, REGISTRY_FILE):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.project_name = project_name
        self.BASE_FOLDER = BASE_FOLDER
        self.REGISTRY_FILE = REGISTRY_FILE
    
    def save_dataset(self, dataset, dataset_name, project_name, db_connection=None, embeddings_df=None):
        """
        Save a dataset or database connection to the project folder and update the registry.

        Args:
            dataset: The dataset to save (can be a file or DataFrame).
            dataset_name (str): The name of the dataset or connection file.
            project_name (str): The project folder name.
            db_connection (dict): Optional database connection details.
        """
        project_folder = Path(self.BASE_FOLDER) / project_name

        try:
            # Create project folder if it doesn't exist
            project_folder.mkdir(parents=True, exist_ok=True)

            # Check if it's a database connection
            if db_connection:
                # Save connection details to a file
                connection_file = project_folder / f"{dataset_name}_connection.json"
                if connection_file.exists():
                    return f"Error: A connection file with the name {dataset_name}_connection.json already exists."

                # Save connection details as JSON
                pd.DataFrame([db_connection]).to_json(connection_file, orient='records', lines=True)
                dataset_path = connection_file
            else:
                # Save the dataset file (e.g., CSV)
                file_path = project_folder / dataset_name
                if file_path.exists():
                    return f"Error: A dataset with the name {dataset_name} already exists."

                dataset.to_csv(file_path, index=False)
                embeddings_df.to_csv("description_embeddings.csv", index=False, header=False)
                dataset_path = file_path

            # Update registry file
            if self.REGISTRY_FILE.exists():
                registry_df = pd.read_csv(self.REGISTRY_FILE)
                next_id = registry_df["id"].max() + 1 if not registry_df.empty else 1
            else:
                registry_df = pd.DataFrame(columns=["id", "dataset_name", "original_location", "project_name", "last_update", "time_of_creation", "is_database"])
                next_id = 1

            new_entry = {
                "id": next_id,
                "dataset_name": dataset_name,
                "original_location": str(dataset_path),
                "project_name": project_name,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "time_of_creation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_database": bool(db_connection)
            }
            registry_df = pd.concat([registry_df, pd.DataFrame([new_entry])], ignore_index=True)
            registry_df.to_csv(self.REGISTRY_FILE, index=False)

            return f"Dataset or connection saved successfully as {dataset_path}."
        except Exception as e:
            return f"Error: {e}"

    def remove_dataset(self, project_to_remove, dataset_to_remove):
        """
        Remove a dataset or database connection file and update the registry.
        """
        project_folder = Path(self.BASE_FOLDER) / project_to_remove
        dataset_path = project_folder / dataset_to_remove

        try:
            if dataset_path.exists():
                os.remove(dataset_path)
                if self.REGISTRY_FILE.exists():
                    registry_df = pd.read_csv(self.REGISTRY_FILE)
                    registry_df = registry_df[~((registry_df["dataset_name"] == dataset_to_remove) & 
                                                (registry_df["project_name"] == project_to_remove))]
                    registry_df.to_csv(self.REGISTRY_FILE, index=False)
                return f"Dataset {dataset_to_remove} removed successfully from project {project_to_remove}."
            else:
                return f"Error: Dataset {dataset_to_remove} not found in project {project_to_remove}."
        except Exception as e:
            return f"Error: {e}"
    
    def get_existing_projects(self):
        """
        Retrieve a list of existing projects (folders) in the base directory.

        Returns:
            list: A list of project folder names found in the base directory.
        """
        project_folders = [folder.name for folder in Path(self.BASE_FOLDER).iterdir() if folder.is_dir()]
        return project_folders

    def get_datasets_in_project(self, project_to_remove):
        """
        Retrieve a list of dataset files within the specified project folder.

        Returns:
            list: A list of dataset file names in the project folder. If the folder does
                  not exist, an empty list is returned.
        """
        project_folder = Path(self.BASE_FOLDER) / project_to_remove
        if project_folder.exists():
            return [file.name for file in project_folder.iterdir() if file.is_file()]
        return []