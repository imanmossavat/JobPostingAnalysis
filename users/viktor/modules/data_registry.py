import os
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from interfaces import IDatasetRegistry

class DatasetRegistry(IDatasetRegistry):
    """
    A class to manage datasets in a structured way. It handles saving datasets,
    removing datasets, retrieving existing projects, and listing datasets in a project.

    Attributes:
        dataset: The dataset file object to be saved or removed.
        project_name (str): The name of the project folder.
        dataset_name (str): The name of the dataset file.
        BASE_FOLDER (str): The base directory where projects and datasets are stored.
        REGISTRY_FILE (Path): Path to the registry file (CSV) for tracking datasets.
    """

    def __init__(self, dataset, project_name, dataset_name, BASE_FOLDER, REGISTRY_FILE):
        """
        Initialize the DatasetRegistry class.

        Args:
            dataset: The dataset file object to be saved or removed.
            project_name (str): The name of the project folder.
            dataset_name (str): The name of the dataset file.
            BASE_FOLDER (str): The base directory where projects and datasets are stored.
            REGISTRY_FILE (Path): Path to the registry file (CSV) for tracking datasets.
        """
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.project_name = project_name
        self.BASE_FOLDER = BASE_FOLDER
        self.REGISTRY_FILE = REGISTRY_FILE

    def save_dataset(self, dataset, project_name):
        """
        Save a dataset to the specified project folder and update the registry file.

        This method creates the project folder if it doesn't exist, saves the uploaded dataset
        file into the folder, and updates the registry CSV file with metadata such as
        dataset name, location, project name, and timestamps.

        Returns:
            str: A success message if the dataset is saved successfully, or an error message
                 if something goes wrong.
        """
        project_folder = Path(self.BASE_FOLDER) / project_name

        try:
            # Create the project folder if it doesn't exist
            project_folder.mkdir(parents=True, exist_ok=True)

            # Check if the dataset already exists
            file_path = project_folder / dataset.name
            if file_path.exists():
                return f"Error: A dataset with the name {dataset.name} already exists."

            # Save the dataset
            with open(file_path, "wb") as f:
                shutil.copyfileobj(dataset, f)

            # Prepare data for the registry entry
            dataset_name = dataset.name
            original_location = str(file_path)
            last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_of_creation = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Load the registry file or create a new one
            if self.REGISTRY_FILE.exists():
                registry_df = pd.read_csv(self.REGISTRY_FILE)
                next_id = registry_df["id"].max() + 1 if not registry_df.empty else 1
            else:
                registry_df = pd.DataFrame(columns=["id", "dataset_name", "original_location", "project_name", "last_update", "time_of_creation"])
                next_id = 1

            # Add the new entry
            new_entry = {
                "id": next_id,
                "dataset_name": dataset_name,
                "original_location": original_location,
                "project_name": self.project_name,
                "last_update": last_update,
                "time_of_creation": time_of_creation
            }
            registry_df = pd.concat([registry_df, pd.DataFrame([new_entry])], ignore_index=True)
            registry_df.to_csv(self.REGISTRY_FILE, index=False)

            return f"Dataset saved successfully in {file_path} and registered."
        except Exception as e:
            return f"Error: {e}"

    ''' def remove_dataset(self, project_to_remove, dataset_to_remove):
        """
        Remove a dataset file from the project folder and delete its registry entry.

        This method checks if the dataset file exists in the specified project folder,
        removes the file, and updates the registry CSV file to exclude the dataset entry.

        Returns:
            str: A success message if the dataset is removed successfully, or an error message
                 if the dataset file is not found or another issue occurs.
        """
        project_folder = Path(self.BASE_FOLDER) / project_to_remove
        dataset_path = project_folder / dataset_to_remove

        try:
            if dataset_path.exists():
                # Remove the dataset file
                os.remove(dataset_path)

                # Remove the entry from the registry
                if self.REGISTRY_FILE.exists():
                    registry_df = pd.read_csv(self.REGISTRY_FILE)
                    registry_df = registry_df[(registry_df["dataset_name"] != self.dataset_name) | (registry_df["project_name"] != self.project_name)]
                    registry_df.to_csv(self.REGISTRY_FILE, index=False)

                return f"Dataset {self.dataset_name} removed successfully."
            else:
                return f"Error: Dataset {self.dataset_name} not found."
        except Exception as e:
            return f"Error: {e}" '''
    
    def remove_dataset(self, project_to_remove, dataset_to_remove):
        """
        Remove a dataset file from the project folder and delete its registry entry.

        Args:
            project_to_remove (str): The name of the project folder.
            dataset_to_remove (str): The name of the dataset file.

        Returns:
            str: Success message or error message.
        """
        project_folder = Path(self.BASE_FOLDER) / project_to_remove
        dataset_path = project_folder / dataset_to_remove

        try:
            if dataset_path.exists():
                # Remove the dataset file
                os.remove(dataset_path)

                # Update the registry file
                if self.REGISTRY_FILE.exists():
                    registry_df = pd.read_csv(self.REGISTRY_FILE)

                    # Filter out the specific entry
                    registry_df = registry_df[~((registry_df["dataset_name"] == dataset_to_remove) & 
                                            (registry_df["project_name"] == project_to_remove))]

                    # Save the updated registry
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