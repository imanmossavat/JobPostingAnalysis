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
    
    def save_dataset(self, dataset, embeddings_dataset, dataset_name, project_name, db_connection=None):
        """
        Save a dataset or database connection to the project folder and update the registry.

        Args:
            dataset: The dataset to save (can be a file or DataFrame).
            dataset_name (str): The name of the dataset or connection file.
            project_name (str): The project folder name.
            db_connection (dict): Optional database connection details.
            embeddings_df (pd.DataFrame): Optional embeddings DataFrame to save.
            embeddings_name (str): Optional name for the embeddings file.
        """
        project_folder = Path(self.BASE_FOLDER) / project_name
        dataset_folder = project_folder / dataset_name

        try:
            # Create project and dataset folders if they don't exist
            dataset_folder.mkdir(parents=True, exist_ok=True)

            # Save database connection
            if db_connection:
                connection_file = dataset_folder / f"{dataset_name}_connection.json"
                if connection_file.exists():
                    return f"Error: A connection file with the name {dataset_name}_connection.json already exists."

                pd.DataFrame([db_connection]).to_json(connection_file, orient='records', lines=True)
                dataset_path = connection_file
            else:
                # Save the dataset
                file_path = dataset_folder / f"{dataset_name}.csv"
                if file_path.exists():
                    return f"Error: A dataset with the name {dataset_name}.csv already exists."
                
                embeddings_file_path = dataset_folder / "embeddings.csv"
                if embeddings_file_path.exists():
                    return f"Error: A dataset with the name {embeddings_file_path}.csv already exists."

                dataset.to_csv(file_path, index=False)
                embeddings_dataset.to_csv(embeddings_file_path, index=False)

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

            return f"Dataset or connection saved successfully in {dataset_folder}."
        except Exception as e:
            return f"Error: {e}. Ensure all inputs are valid and directories are writable."

    def remove_dataset(self, project_to_remove, dataset_to_remove):
        """
        Remove a dataset or database connection subfolder and update the registry.

        Args:
            project_to_remove (str): The name of the project containing the dataset.
            dataset_to_remove (str): The name of the dataset (subfolder) to remove.

        Returns:
            str: A message indicating success or the error encountered.
        """
        project_folder = Path(self.BASE_FOLDER) / project_to_remove
        dataset_folder = project_folder / dataset_to_remove

        try:
            # Check if the dataset subfolder exists
            if dataset_folder.exists() and dataset_folder.is_dir():
                # Delete all files in the subfolder and then remove the folder
                for file in dataset_folder.iterdir():
                    if file.is_file():
                        file.unlink()  # Remove individual files
                    elif file.is_dir():
                        # Recursively delete subdirectories
                        for subfile in file.iterdir():
                            subfile.unlink()
                        file.rmdir()
                dataset_folder.rmdir()  # Remove the main dataset folder

                # Update the registry
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

    def get_datasets_in_project(self, project_name):
        """
        Retrieve a list of dataset subfolders within the specified project folder.

        Args:
            project_name (str): The name of the project folder.

        Returns:
            list: A list of dataset subfolder names in the project folder. If the folder
                  does not exist, an empty list is returned.
        """
        project_folder = Path(self.BASE_FOLDER) / project_name
        if project_folder.exists() and project_folder.is_dir():
            return [folder.name for folder in project_folder.iterdir() if folder.is_dir()]
        return []