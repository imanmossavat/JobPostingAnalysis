from modules import DatasetRegistry
from config import Config

# Load configuration
configs = Config()

base_folder = configs.base_folder
registry_file = configs.registry_file

class DataRegistryManager():
    """
    A class that provides a wrapper around the DatasetRegistry to manage datasets
    within a project, including saving, removing, and querying datasets.

    Attributes:
        project_name (str): The name of the project associated with the dataset.
        dataset (Dataset): The dataset object to be managed.
        dataset_registry (DatasetRegistry): An instance of DatasetRegistry that handles dataset operations.
    """
    
    def __init__(self, dataset, file_name, project_name):
        """
        Initializes the DataRegistryManager with the provided dataset and project name.

        Args:
            dataset (Dataset): The dataset object that will be managed.
            project_name (str): The name of the project to associate with the dataset.
        """
        self.project_name = project_name
        self.dataset = dataset
        self.file_name = file_name
        self.dataset_registry = DatasetRegistry(dataset, project_name, file_name, base_folder, registry_file)
    
    def save_dataset(self, dataset, project_name):
        """
        Saves the current dataset to the registry. This is a wrapper around the 
        DatasetRegistry.save_dataset method to integrate it into a Streamlit app.

        Returns:
            bool: True if the dataset was successfully saved, otherwise False.
        """
        return self.dataset_registry.save_dataset(dataset, project_name)

    def remove_dataset(self, project_to_remove, dataset_to_remove):
        """
        Removes the current dataset from the registry. This is a wrapper around the 
        DatasetRegistry.remove_dataset method to integrate it into a Streamlit app.

        Returns:
            bool: True if the dataset was successfully removed, otherwise False.
        """
        return self.dataset_registry.remove_dataset(project_to_remove, dataset_to_remove)

    def get_existing_projects(self):
        """
        Retrieves all existing projects in the dataset registry. This is a wrapper 
        around the DatasetRegistry.get_existing_projects method to use in the Streamlit app.

        Returns:
            list: A list of project names that are already present in the registry.
        """
        return self.dataset_registry.get_existing_projects()

    def get_datasets_in_project(self, project_to_remove):
        """
        Retrieves all datasets associated with the current project. This is a wrapper 
        around the DatasetRegistry.get_datasets_in_project method to use in the Streamlit app.

        Returns:
            list: A list of datasets currently registered under the specified project.
        """
        return self.dataset_registry.get_datasets_in_project(project_to_remove)