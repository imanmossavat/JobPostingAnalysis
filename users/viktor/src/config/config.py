from pathlib import Path
from .files import english_dataset_path, dutch_dataset_path
from .files import industries_and_sectors, job_titles_clusters
from .files import stopword_file_names
from .files import reports_folder_path
from .files import registry_file_path
from .files import registry_folder_path
from .files import keywords_folder_path

# from pathlib import Path

class Config():
    def __init__(self):
        self._n_topics = 5
        self._num_top_words = 10
        self._epochs = 50

        self._csv_dataset = english_dataset_path

        self._keywords_folder_path = keywords_folder_path

        self._topics_file = industries_and_sectors

        self._stopword_file_names = stopword_file_names

        self._name_of_topics = "Job Area"

        self._text_column = "description"

        self._reports_folder_path = reports_folder_path

        self._base_folder = registry_folder_path

        self._registry_file = registry_file_path

        # Column renames

        self._COLUMN_RENAMES = {
            'Id': 'job_id',
            'Description': 'description',
            'Title': 'title',
            'Language': 'language',
            'CreatedAt': 'original_listed_time',
            'Compensation': 'med_salary_monthly',
            'med_salary': 'med_salary_monthly',
            'AddressId': 'location'
        }

        # Define special handling for 'AddressId' -> 'location' to create 'company_name'
        self._SPECIAL_HANDLINGS_COLUMNS = {
            'AddressId': 'company_name'
        }

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
    def keywords_folder_path(self):
        return self._keywords_folder_path
    
    @property
    def base_folder(self):
        return self._base_folder
    
    @property
    def registry_file(self):
        return self._registry_file
    
    @property
    def COLUMN_RENAMES(self):
        return self._COLUMN_RENAMES

    @property
    def SPECIAL_HANDLINGS_COLUMNS(self):
        return self._SPECIAL_HANDLINGS_COLUMNS

    # Setters
    @n_topics.setter
    def n_topics(self, value):
        if isinstance(value, int) and value > 0:
            self._n_topics = value
        else:
            raise ValueError("n_topics must be a positive integer.")

    @num_top_words.setter
    def num_top_words(self, value):
        if isinstance(value, int) and value > 0:
            self._num_top_words = value
        else:
            raise ValueError("num_top_words must be a positive integer.")

    @epochs.setter
    def epochs(self, value):
        if isinstance(value, int) and value > 0:
            self._epochs = value
        else:
            raise ValueError("epochs must be a positive integer.")

    @csv_dataset.setter
    def csv_dataset(self, value):
        if isinstance(value, str):
            self._csv_dataset = value
        else:
            raise ValueError("csv_dataset must be a string.")

    @topics_file.setter
    def topics_file(self, value):
        if isinstance(value, str):
            self._topics_file = value
        else:
            raise ValueError("topics_file must be a string.")

    @reports_folder_path.setter
    def reports_folder_path(self, value):
        if isinstance(value, str):
            self._reports_folder_path = value
        else:
            raise ValueError("reports_folder_path must be a string.")
    
    @keywords_folder_path.setter
    def keywords_folder_path(self, value):
        if isinstance(value, str):
            self._keywords_folder_path = value
        else:
            raise ValueError("keywords_folder_path must be a string.")

    @name_of_topics.setter
    def name_of_topics(self, value):
        if isinstance(value, str):
            self._name_of_topics = value
        else:
            raise ValueError("name_of_topics must be a string.")

    @stopword_file_names.setter
    def stopword_file_names(self, value):
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            self._stopword_file_names = value
        else:
            raise ValueError("stopword_file_names must be a list of strings.")
        
    @text_column.setter
    def text_column(self, value):
        if isinstance(value, str):
            self._text_column = value
        else:
            raise ValueError("text_column must be a string.")
        
    @base_folder.setter
    def base_folder(self, value):
        if isinstance(value, str):
            self._base_folder = value
        else:
            raise ValueError("base_folder must be a string.")
        
    @registry_file.setter
    def registry_file(self, value):
        if isinstance(value, str):
            self._registry_file = Path(value)  # Convert string to Path
        elif isinstance(value, Path):
            self._registry_file = value  # Directly accept Path objects
        else:
            raise ValueError("registry_file must be a string or Path object.")
        
    @COLUMN_RENAMES.setter
    def COLUMN_RENAMES(self, value):
        if isinstance(value, dict):
            # Ensure that the dictionary has valid keys and values
            if all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                self._COLUMN_RENAMES = value
            else:
                raise ValueError("COLUMN_RENAMES must be a dictionary with string keys and values.")
        else:
            raise ValueError("COLUMN_RENAMES must be a dictionary.")

    @SPECIAL_HANDLINGS_COLUMNS.setter
    def SPECIAL_HANDLINGS_COLUMNS(self, value):
        if isinstance(value, dict):
            # Ensure that the dictionary has valid keys and values
            if all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                self._SPECIAL_HANDLINGS_COLUMNS = value
            else:
                raise ValueError("SPECIAL_HANDLINGS_COLUMNS must be a dictionary with string keys and values.")
        else:
            raise ValueError("SPECIAL_HANDLINGS_COLUMNS must be a dictionary.")

    # Method to return the topic modeling input variables
    def topic_modeling_input_variables(self):
        return self._n_topics, self._num_top_words, self._epochs