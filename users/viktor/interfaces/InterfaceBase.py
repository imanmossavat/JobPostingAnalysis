from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict
import numpy as np

class IWordCloudGenerator(ABC):
    """
    Interface for WordCloudGenerator classes.
    """

    @abstractmethod
    def __init__(self, 
                 df: pd.DataFrame, 
                 keyword_dict: Dict[str, List[str]], 
                 output_folder: str, 
                 name_of_topics: str, 
                 stopword_files: List[str], 
                 column: str = 'Description'):
        """
        Initialize the WordCloudGenerator.

        Parameters:
        df (pd.DataFrame): DataFrame containing job descriptions and other relevant data.
        keyword_dict (dict): Dictionary where keys are topic names and values are lists of associated keywords.
        output_folder (str): Folder where the generated WordCloud images will be saved.
        name_of_topics (str): Common title for the WordCloud topics.
        stopword_files (list): List of paths to stopword files for text preprocessing.
        column (str): Name of the column containing job descriptions in the DataFrame.
        """
        pass

    @abstractmethod
    def generate_wordcloud_for_topic(self) -> List[str]:
        """
        Generate WordCloud images for each topic based on keyword frequencies in job descriptions.

        Returns:
        list: Paths to the generated WordCloud images.
        """
        pass

class ITextPreprocessor(ABC):
    """
    Abstract base class for text preprocessing.
    
    This class defines the interface for preprocessing a list of text data, such as cleaning,
    tokenization, and transformation into a format suitable for further analysis or modeling.
    """
    
    @abstractmethod
    def preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of text data by applying necessary text-cleaning operations.
        
        Args:
            texts (List[str]): A list of text strings to be processed.

        Returns:
            List[str]: A list of preprocessed text strings.
        """
        pass

class IDatasetRegistry(ABC):
    """
    Interface for IDatasetRegistry class, defining the methods that should be implemented
    to manage datasets in a structured way.
    """

    @abstractmethod
    def __init__(self, dataset, project_name, dataset_name, BASE_FOLDER, REGISTRY_FILE):
        """
        Initialize the IDatasetRegistry class.

        Args:
            dataset: The dataset file object to be saved or removed.
            project_name (str): The name of the project folder.
            dataset_name (str): The name of the dataset file.
            BASE_FOLDER (str): The base directory where projects and datasets are stored.
            REGISTRY_FILE (Path): Path to the registry file (CSV) for tracking datasets.
        """
        pass

    @abstractmethod
    def save_dataset(self) -> str:
        """
        Save a dataset to the specified project folder and update the registry file.

        Returns:
            str: Success or error message.
        """
        pass

    @abstractmethod
    def remove_dataset(self) -> str:
        """
        Remove a dataset file from the project folder and delete its registry entry.

        Returns:
            str: Success or error message.
        """
        pass

    @abstractmethod
    def get_existing_projects(self) -> List[str]:
        """
        Retrieve a list of existing projects (folders) in the base directory.

        Returns:
            list: A list of project folder names found in the base directory.
        """
        pass

    @abstractmethod
    def get_datasets_in_project(self) -> List[str]:
        """
        Retrieve a list of dataset files within the specified project folder.

        Returns:
            list: A list of dataset file names in the project folder.
        """
        pass

class IDataFormatter(ABC):
    """
    Interface for DataFormatter. This defines the methods required for 
    formatting and transforming a DataFrame.
    """

    @abstractmethod
    def __init__(self, df: pd.DataFrame, column_renames: dict, special_handlings_columns: dict):
        """
        Initializes the DataFormatter with a DataFrame.

        Args:
            df (pd.DataFrame): The dataset to be formatted.
            column_renames (dict): Dictionary mapping old column names to new names.
            special_handlings_columns (dict): Dictionary of columns that require special handling.
        """
        self.df = df
        self.column_renames = column_renames
        self.special_handlings_columns = special_handlings_columns

    @abstractmethod
    def rename_columns(self) -> pd.DataFrame:
        """
        Renames specific columns in the DataFrame and adds new columns where applicable.
        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        pass

class IFeatureExtractor(ABC):
    """
    Interface for feature extraction implementations.
    """

    @abstractmethod
    def __init__(self, column: str, keyword_dict: Dict[str, list], temp: float = 1.0):
        """
        Initialize the feature extractor.

        Args:
            column (str): Name of the column containing text data.
            keyword_dict (dict): Dictionary mapping feature names to lists of keywords.
            temp (float): Temperature parameter for normalization (optional).
        """
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing text data.

        Returns:
            pd.DataFrame: A DataFrame with extracted features.
        """
        pass

class ISemiannualFeatureDistribution(ABC):
    """
    Abstract base class for plotting trends.

    Methods:
        plot_trends(trend_df: pd.DataFrame, monthly_trend_df: pd.DataFrame) -> str:
            Generate a trend plot and return the file path of the saved plot.
    """
    @abstractmethod
    def plot_trends(self, trend_df: pd.DataFrame, monthly_trend_df: pd.DataFrame) -> str:
        """
        Abstract method to plot trends based on input data.

        Args:
            trend_df (pd.DataFrame): DataFrame containing the trend data.
            monthly_trend_df (pd.DataFrame): DataFrame containing monthly trend adjustments.

        Returns:
            str: File path of the saved plot.
        """
        pass

class ISoftmaxTransformer(ABC):
    """
    Interface for a class that applies a softmax transformation with a temperature parameter.
    """
    
    @abstractmethod
    def apply(self, row: np.ndarray) -> np.ndarray:
        """
        Applies softmax transformation to the given row.
        
        Args:
            row: A numpy array representing the row of features to be transformed.
        
        Returns:
            A numpy array after applying softmax.
        """
        pass

    @abstractmethod
    def set_temperature(self, temp: float):
        """
        Sets the temperature for the softmax operation.
        
        Args:
            temp: The temperature to be used in softmax calculation.
        """
        pass

class IKeywordFeatureExtractor(ABC):
    def __init__(self, df: pd.DataFrame, column: str, keyword_dict: dict, temp: float = 0.5):
        """
        Initialize the Keyword Feature Extractor interface.

        Args:
            df (pd.DataFrame): DataFrame containing text data.
            column (str): Column name in the DataFrame containing the text.
            keyword_dict (dict): A dictionary mapping feature names to keyword lists.
            temp (float, optional): Temperature value for softmax scaling (default is 0.5).
        """
        self.df = df
        self.column = column
        self.keyword_dict = keyword_dict
        self.temp = temp

    @abstractmethod
    def extract_features(self) -> pd.DataFrame:
        """
        Abstract method to extract features from the DataFrame based on the keyword dictionary.

        Returns:
            pd.DataFrame: DataFrame with new features.
        """
        pass


class IBoxPlots(ABC):
    def __init__(self, trend_df: pd.DataFrame, monthly_trend_df: pd.DataFrame, role_columns: list, 
                 output_subfolder_base: str, reports_folder_path: str):
        """
        Initialize the BoxPlots interface.

        Args:
            trend_df (pd.DataFrame): DataFrame with trend data.
            monthly_trend_df (pd.DataFrame): DataFrame with monthly trend data.
            role_columns (list): List of columns (features) to visualize.
            output_subfolder_base (str): Base name for the output subfolder.
            reports_folder_path (str): Path to store the report files.
        """
        self.trend_df = trend_df
        self.monthly_trend_df = monthly_trend_df
        self.role_columns = role_columns
        self.output_subfolder_base = output_subfolder_base
        self.reports_folder_path = reports_folder_path

    @abstractmethod
    def plot_distribution(self):
        """
        Abstract method to create and save box plots based on the trends.

        Returns:
            None: Saves box plot visualizations to the output folder.
        """
        pass

class ITopicOverlapGraphGenerator(ABC):
    """
    Interface for generating a topic overlap graph.
    """
    
    @abstractmethod
    def generate_graph(self, keyword_dict: dict, output_subfolder: str, name_of_topics: str, filtered_df: pd.DataFrame = None) -> None:
        """
        Generate and save a topic overlap graph based on keyword overlaps.
        
        Args:
            keyword_dict: Dictionary where keys are topics and values are lists of keywords.
            output_subfolder: Folder to save the graph.
            name_of_topics: Name of the topics for the graph's title.
            filtered_df: Optional filtered DataFrame for the graph.
        
        Returns:
            None
        """
        pass

class ITopicAssignment(ABC):
    """
    Interface for assigning the most likely topic based on a given text and a set of keywords.
    """

    @abstractmethod
    def assign_most_likely_topic(self, df: pd.DataFrame, column: str, keyword_dict: dict) -> pd.DataFrame:
        """
        Assign the most likely topic to each row in the given DataFrame based on keyword matching.

        Args:
            df: DataFrame containing the text data.
            column: Column name containing the text to analyze.
            keyword_dict: Dictionary where keys are topic names and values are lists of keywords.

        Returns:
            DataFrame: The input DataFrame with an additional column 'Most_Likely_Topic'.
        """
        pass

class ITopicModel(ABC):
    """
    Interface for Topic Modeling classes.
    """

    @abstractmethod
    def __init__(self, n_topics, data, stopword_files, num_top_words, epochs, output_subfolder):
        """
        Initialize the topic model with necessary parameters.

        Args:
            n_topics (int): Number of topics to extract.
            data (dict): Dataset containing the descriptions to process.
            stopword_files (list): File paths to stopword lists.
            num_top_words (int): Number of top words per topic to display.
            epochs (int): Maximum number of iterations for the model.
            output_subfolder (str): Directory for saving output files.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Fit the model to the data.
        """
        pass

    @abstractmethod
    def display_topics(self):
        """
        Display the top words for each topic.
        """
        pass

    @abstractmethod
    def calculate_topic_coherence(self):
        """
        Calculate the topic coherence score.
        """
        pass

    @abstractmethod
    def calculate_topic_diversity(self):
        """
        Calculate the topic diversity score.
        """
        pass

    @abstractmethod
    def calculate_silhouette_score(self):
        """
        Calculate the silhouette score for clustering.
        """
        pass

    @abstractmethod
    def evaluate_clustering_stability(self, num_runs):
        """
        Evaluate clustering stability by running multiple trials.
        """
        pass

    @abstractmethod
    def calculate_cosine_similarity(self):
        """
        Calculate pairwise cosine similarity between topics.
        """
        pass

    @abstractmethod
    def calculate_topic_percentage(self):
        """
        Calculate the percentage contribution of each topic.
        """
        pass

class ISkillKnowledgeExtractor(ABC):
    """
    Interface for extracting skills and knowledge from job descriptions, saving progress, and generating reports.
    """

    @abstractmethod
    def extract_skills_and_knowledge(self, text: str, lang: str) -> dict:
        """
        Extract skills and knowledge entities from the given text using pre-trained models.

        Args:
            text (str): The text from which to extract skills and knowledge.
            lang (str): The language of the text (used to adjust processing if necessary).

        Returns:
            dict: A dictionary containing the text, detected skills, detected knowledge, and the detected language.
        """
        pass

    @abstractmethod
    def save_progress(self, progress_data: pd.DataFrame, output_subfolder: str) -> None:
        """
        Save intermediate results to CSV to track progress.

        Args:
            progress_data (pd.DataFrame): The current progress data to save.
            output_subfolder (str): Path where the progress file should be saved.
        """
        pass

    @abstractmethod
    def generate_report(self, input_file: str, output_file: str, output_subfolder: str) -> None:
        """
        Generate a report in Word document format summarizing the experiment's progress.

        Args:
            input_file (str): The input file path.
            output_file (str): The output file path.
            output_subfolder (str): Path to save the generated report.
        """
        pass

class IWord2VecEmbeddingTrendAnalysis(ABC):
    """
    Abstract base class for Word2Vec embedding trend analysis.
    This class defines the methods necessary for preprocessing text, 
    training Word2Vec models, tracking trends over time, and visualizing the results.
    """

    @abstractmethod
    def preprocess(self, text: str) -> list:
        """
        Preprocesses the given text by tokenizing, lemmatizing, and removing stopwords.
        
        Args:
            text (str): The text to preprocess.
        
        Returns:
            list: A list of processed words.
        """
        pass

    @abstractmethod
    def tokenize_and_train(self):
        """
        Tokenizes the text and trains the Word2Vec model on the tokenized data.
        
        This method should handle the training of the Word2Vec model using the provided data.
        The model should be stored after training for later use.
        """
        pass

    @abstractmethod
    def produce_scatter_plot(self):
        """
        Produces a scatter plot visualizing trends of keywords over time.
        
        This method tracks specific keywords, aggregates trends over time, and visualizes
        the trends as a scatter plot, which may include smoothed lines using Gaussian Process
        regression and the marking of significant events (e.g., COVID-19 lockdown, ChatGPT launch).
        """
        pass

    @abstractmethod
    def track_trends2(self):
        """
        Tracks trends over time and generates a plot with smoothed keyword trends.
        
        This method tracks a predefined set of keywords, generates trend data over time,
        applies Gaussian Process regression for smoothing, and produces a line plot of these trends.
        Events such as COVID-19 and ChatGPT launch may also be visualized on the plot.
        """
        pass

    @abstractmethod
    def track_trends3(self):
        """
        Tracks trends over time with a more detailed plot, including ruptures analysis.
        
        Similar to `track_trends2`, but this method includes detection of changes or 
        ruptures in trends using the `ruptures` library. The plot is more detailed and shows
        separate subplots for each keyword, along with rupture lines and event markers.
        """
        pass