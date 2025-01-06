import os
import sys
import pytest
import pandas as pd
import unittest
from datetime import datetime
from pathlib import Path

# Add the src folder to the Python path
current_dir = os.path.dirname(__file__)

# Traverse up the directory structure to find 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../src'))

# Add 'src' to the Python path
sys.path.insert(0, src_dir)

from modules import DatasetRegistry  # Make sure to import the class correctly

# Constants for testing
BASE_FOLDER = './test_base_folder'
REGISTRY_FILE = Path(BASE_FOLDER) / 'registry.csv'

# Utility function to create a temporary project folder
def setup_module():
    if not os.path.exists(BASE_FOLDER):
        os.makedirs(BASE_FOLDER)

def teardown_module():
    if os.path.exists(BASE_FOLDER):
        # Iterate over the directory to remove files and subdirectories
        for folder in Path(BASE_FOLDER).iterdir():
            if folder.is_dir():
                for file in folder.iterdir():
                    file.unlink()  # Remove each file
                folder.rmdir()  # Remove the subdirectory
            else:
                folder.unlink()  # If it's a file, remove it
        os.rmdir(BASE_FOLDER)  # Finally, remove the base folder

# Unit tests using pytest
@pytest.fixture
def dataset_registry():
    return DatasetRegistry(None, "test_project", "test_dataset", BASE_FOLDER, REGISTRY_FILE)

def test_save_dataset_new(dataset_registry):
    # Create a simple DataFrame
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    
    # Save dataset
    result = dataset_registry.save_dataset(df, "test_dataset.csv", "test_project")
    
    assert "Dataset or connection saved successfully" in result
    assert (Path(BASE_FOLDER) / "test_project" / "test_dataset.csv").exists()

def test_save_existing_dataset(dataset_registry):
    # Create a simple DataFrame
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    
    # Save dataset first time
    dataset_registry.save_dataset(df, "test_dataset.csv", "test_project")
    
    # Try saving the same dataset again
    result = dataset_registry.save_dataset(df, "test_dataset.csv", "test_project")
    
    assert "Error: A dataset with the name test_dataset.csv already exists." in result

def test_save_connection(dataset_registry):
    db_connection = {"host": "localhost", "port": 5432, "dbname": "test_db"}
    
    # Ensure that the connection file does not exist
    connection_file = Path(BASE_FOLDER) / "test_project" / "test_db_connection_connection.json"
    if connection_file.exists():
        connection_file.unlink()

    # Save database connection
    result = dataset_registry.save_dataset(None, "test_db_connection", "test_project", db_connection=db_connection)
    
    assert "Dataset or connection saved successfully" in result
    assert (Path(BASE_FOLDER) / "test_project" / "test_db_connection_connection.json").exists()

def test_save_connection_unique_filename(dataset_registry):
    db_connection = {"host": "localhost", "port": 5432, "dbname": "test_db"}

    # Use a unique connection name to avoid conflict
    connection_name = f"test_db_connection_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Save database connection
    result = dataset_registry.save_dataset(None, connection_name, "test_project", db_connection=db_connection)
    
    assert "Dataset or connection saved successfully" in result
    assert (Path(BASE_FOLDER) / "test_project" / f"{connection_name}_connection.json").exists()

def test_remove_dataset(dataset_registry):
    # Create a simple DataFrame
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    
    # Save dataset
    dataset_registry.save_dataset(df, "test_dataset.csv", "test_project")
    
    # Remove dataset
    result = dataset_registry.remove_dataset("test_project", "test_dataset.csv")
    
    assert "Dataset test_dataset.csv removed successfully" in result
    assert not (Path(BASE_FOLDER) / "test_project" / "test_dataset.csv").exists()

def test_remove_non_existing_dataset(dataset_registry):
    result = dataset_registry.remove_dataset("test_project", "non_existing_dataset.csv")
    assert "Error: Dataset non_existing_dataset.csv not found in project test_project." in result

def test_get_existing_projects(dataset_registry):
    # Create a project folder manually for testing
    project_folder = Path(BASE_FOLDER) / "existing_project"
    project_folder.mkdir(parents=True, exist_ok=True)
    
    # Retrieve existing projects
    projects = dataset_registry.get_existing_projects()
    
    assert "existing_project" in projects
    project_folder.rmdir()

def test_get_datasets_in_project(dataset_registry):
    # Create a project folder and dataset file manually for testing
    project_folder = Path(BASE_FOLDER) / "existing_project"
    project_folder.mkdir(parents=True, exist_ok=True)
    dataset_file = project_folder / "existing_dataset.csv"
    dataset_file.write_text("col1,col2\n1,2\n3,4\n")
    
    datasets = dataset_registry.get_datasets_in_project("existing_project")
    
    assert "existing_dataset.csv" in datasets
    dataset_file.unlink()
    project_folder.rmdir()

# Integration tests using unittest
import unittest

class TestDatasetRegistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        setup_module()

    @classmethod
    def tearDownClass(cls):
        teardown_module()

    def setUp(self):
        self.registry = DatasetRegistry(None, "test_project", "test_dataset", BASE_FOLDER, REGISTRY_FILE)

    def test_save_and_remove_dataset(self):
        # Create a DataFrame and save it
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        save_result = self.registry.save_dataset(df, "test_dataset.csv", "test_project")
        self.assertIn("Dataset or connection saved successfully", save_result)
        self.assertTrue((Path(BASE_FOLDER) / "test_project" / "test_dataset.csv").exists())

        # Remove the dataset
        remove_result = self.registry.remove_dataset("test_project", "test_dataset.csv")
        self.assertIn("Dataset test_dataset.csv removed successfully", remove_result)
        self.assertFalse((Path(BASE_FOLDER) / "test_project" / "test_dataset.csv").exists())

    def test_save_connection_and_registry_update(self):
        db_connection = {"host": "localhost", "port": 5432, "dbname": "test_db"}
        
        # Ensure that the connection file does not exist before saving
        connection_file = Path(BASE_FOLDER) / "test_project" / "test_db_connection_connection.json"
        if connection_file.exists():
            connection_file.unlink()

        save_result = self.registry.save_dataset(None, "test_db_connection", "test_project", db_connection=db_connection)
        self.assertIn("Dataset or connection saved successfully", save_result)
        
        # Check if the connection file is created
        connection_file = Path(BASE_FOLDER) / "test_project" / "test_db_connection_connection.json"
        self.assertTrue(connection_file.exists())
        
        # Check if the registry is updated
        registry_df = pd.read_csv(REGISTRY_FILE)
        self.assertGreater(len(registry_df), 0)

    def test_get_projects_and_datasets(self):
        # Manually create a project and dataset
        project_folder = Path(BASE_FOLDER) / "new_project"
        project_folder.mkdir(parents=True, exist_ok=True)
        dataset_file = project_folder / "new_dataset.csv"
        dataset_file.write_text("col1,col2\n1,2\n3,4\n")
        
        # Test get existing projects
        projects = self.registry.get_existing_projects()
        self.assertIn("new_project", projects)
        
        # Test get datasets in project
        datasets = self.registry.get_datasets_in_project("new_project")
        self.assertIn("new_dataset.csv", datasets)

        # Cleanup
        dataset_file.unlink()
        project_folder.rmdir()

if __name__ == '__main__':
    unittest.main()