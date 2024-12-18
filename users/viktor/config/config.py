from pathlib import Path
from .files import english_dataset_path, dutch_dataset_path
from .files import industries_and_sectors, job_titles_clusters
from .files import stopword_file_names
from .files import reports_folder_path
from .files import registry_file_path
from .files import registry_folder_path

# from pathlib import Path

class Config():
    def __init__(self):
        self._n_topics = 5
        self._num_top_words = 10
        self._epochs = 50

        self._csv_dataset = english_dataset_path

        self._topics_file = job_titles_clusters

        self._stopword_file_names = stopword_file_names

        self._name_of_topics = "Job Area"

        self._text_column = "Description"

        self._reports_folder_path = reports_folder_path

        self._base_folder = registry_folder_path

        # self._registry_file = Path(registry_folder_path) / "registry.csv"

        self._registry_file = registry_file_path

    # Getters
    @property
    def n_topics(self):
        return self._n_topics

    @property
    def num_top_words(self):
        return self._num_top_words

    @property
    def epochs(self):
        return self._epochs

    @property
    def csv_dataset(self):
        return self._csv_dataset

    @property
    def topics_file(self):
        return self._topics_file

    @property
    def name_of_topics(self):
        return self._name_of_topics
    
    @property
    def stopword_file_names(self):
        return self._stopword_file_names

    @property
    def text_column(self):
        return self._text_column
    
    @property
    def reports_folder_path(self):
        return self._reports_folder_path
    
    @property
    def base_folder(self):
        return self._base_folder
    
    @property
    def registry_file(self):
        return self._registry_file

    # Setters
    @n_topics.__setattr__
    def n_topics(self, value):
        if isinstance(value, int) and value > 0:
            self._n_topics = value
        else:
            raise ValueError("n_topics must be a positive integer.")

    @num_top_words.__setattr__
    def num_top_words(self, value):
        if isinstance(value, int) and value > 0:
            self._num_top_words = value
        else:
            raise ValueError("num_top_words must be a positive integer.")

    @epochs.__setattr__
    def epochs(self, value):
        if isinstance(value, int) and value > 0:
            self._epochs = value
        else:
            raise ValueError("epochs must be a positive integer.")

    @csv_dataset.__setattr__
    def csv_dataset(self, value):
        if isinstance(value, str):
            self._csv_dataset = value
        else:
            raise ValueError("csv_dataset must be a string.")

    @topics_file.__setattr__
    def topics_file(self, value):
        if isinstance(value, str):
            self._topics_file = value
        else:
            raise ValueError("topics_file must be a string.")

    @reports_folder_path.__setattr__
    def reports_folder_path(self, value):
        if isinstance(value, str):
            self._reports_folder_path = value
        else:
            raise ValueError("reports_folder_path must be a string.")

    @name_of_topics.__setattr__
    def name_of_topics(self, value):
        if isinstance(value, str):
            self._name_of_topics = value
        else:
            raise ValueError("name_of_topics must be a string.")

    @stopword_file_names.__setattr__
    def stopword_file_names(self, value):
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            self._stopword_file_names = value
        else:
            raise ValueError("stopword_file_names must be a list of strings.")
        
    @text_column.__setattr__
    def text_column(self, value):
        if isinstance(value, str):
            self._text_column = value
        else:
            raise ValueError("text_column must be a string.")
        
    @base_folder.__setattr__
    def base_folder(self, value):
        if isinstance(value, str):
            self._base_folder = value
        else:
            raise ValueError("base_folder must be a string.")
        
    @registry_file.__setattr__
    def registry_file(self, value):
        if isinstance(value, str):
            self._registry_file = Path(value)  # Convert string to Path
        elif isinstance(value, Path):
            self._registry_file = value  # Directly accept Path objects
        else:
            raise ValueError("registry_file must be a string or Path object.")

    # Method to return the topic modeling input variables
    def topic_modeling_input_variables(self):
        return self._n_topics, self._num_top_words, self._epochs