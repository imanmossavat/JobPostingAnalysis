from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict
import numpy as np

class IWordCloudGenerator(ABC):
    """
    Abstract base class for a word cloud generator.
    
    This class defines the interface for generating word clouds based on topic-specific keywords 
    from a DataFrame of text data. The output is a list of strings representing the generated word clouds.
    """
    
    @abstractmethod
    def generate_wordcloud_for_topic(
        self, 
        df: pd.DataFrame, 
        keyword_dict: Dict[str, List[str]], 
        output_folder: str, 
        name_of_topics: str, 
        column: str = 'Description'
    ) -> List[str]:
        """
        Generates word clouds for specific topics based on a keyword dictionary.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.
            keyword_dict (Dict[str, List[str]]): A dictionary where the keys are topic names 
                                                 and the values are lists of keywords associated with each topic.
            output_folder (str): Path to the folder where the word clouds should be saved.
            name_of_topics (str): Name or identifier of the topics for which the word cloud is generated.
            column (str): The column name in the DataFrame containing the text data (default is 'Description').

        Returns:
            List[str]: A list of strings representing the generated word clouds (or paths to saved word clouds).
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

class IFeatureExtractor(ABC):
    """
    Abstract base class for feature extraction.

    Methods:
        extract_features(df: pd.DataFrame) -> pd.DataFrame:
            Extract features from a DataFrame and return the transformed DataFrame.
    """
    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to extract features from a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.

        Returns:
            pd.DataFrame: The transformed DataFrame with extracted features.
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
    @abstractmethod
    def extract_features(self, df: pd.DataFrame, column: str, keyword_dict: Dict[str, List[str]], temp: float = 1.0) -> pd.DataFrame:
        """
        Extract features from a text column based on a keyword dictionary.
        """
        pass

class IBoxPlots(ABC):
    @abstractmethod
    def plot_distribution(self, trend_df: pd.DataFrame, monthly_trend_df: pd.DataFrame, role_columns: List[str], output_subfolder_base: str) -> None:
        """
        Create and save box plots for feature percentage distribution.
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
    Abstract base class for topic modeling.
    
    This class defines the interface for performing topic modeling, which includes fitting a model, 
    displaying extracted topics, and calculating various metrics to evaluate the quality of the topics.
    """
    
    @abstractmethod
    def fit(self):
        """
        Fits the topic model to the data.
        
        This method should implement the process of training a topic modeling algorithm, such as LDA or NMF.
        """
        pass

    @abstractmethod
    def display_topics(self):
        """
        Displays the extracted topics from the fitted model.
        
        This method should output or return the topics discovered by the model in an interpretable format.
        """
        pass

    @abstractmethod
    def calculate_topic_coherence(self) -> float:
        """
        Calculates the topic coherence score for the fitted model.
        
        Topic coherence measures the interpretability and quality of topics based on the co-occurrence of words.
        
        Returns:
            float: A score representing the coherence of the topics.
        """
        pass

    @abstractmethod
    def calculate_topic_diversity(self) -> float:
        """
        Calculates the topic diversity score for the fitted model.
        
        Topic diversity assesses how distinct the topics are from each other.
        
        Returns:
            float: A score representing the diversity of the topics.
        """
        pass

    @abstractmethod
    def calculate_silhouette_score(self) -> float:
        """
        Calculates the silhouette score for clustering the topics.
        
        The silhouette score measures how similar each point is to its own cluster compared to other clusters.
        
        Returns:
            float: A score indicating the quality of clustering.
        """
        pass

    @abstractmethod
    def calculate_cosine_similarity(self) -> float:
        """
        Calculates the cosine similarity between the topics.
        
        Cosine similarity measures how similar the topics are based on their word distributions.
        
        Returns:
            float: A score representing the cosine similarity between topics.
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