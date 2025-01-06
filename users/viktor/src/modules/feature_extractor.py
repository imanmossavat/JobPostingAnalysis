import re
import numpy as np
import pandas as pd
from scipy.special import softmax
from interfaces import IFeatureExtractor

class KeywordFeatureExtractor(IFeatureExtractor):
    """
    Implementation of IFeatureExtractor for keyword-based feature extraction.

    Attributes:
        column (str): Name of the column containing text data.
        keyword_dict (dict): Dictionary of feature names and their associated keywords.
        temp (float): Temperature parameter for softmax normalization.
    """

    def __init__(self, column: str, keyword_dict: dict, temp: float = 1.0):
        """
        Initialize the KeywordFeatureExtractor.

        Args:
            column (str): Name of the text column to process.
            keyword_dict (dict): Dictionary mapping feature names to keyword lists.
            temp (float): Temperature for softmax normalization. Default is 1.0.
        """
        self.column = column
        self.keyword_dict = keyword_dict
        self.temp = temp

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract keyword-based features from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.

        Returns:
            pd.DataFrame: A DataFrame with extracted features and softmax-normalized values.
        """

        feature_df = df.copy()

        # Loop over the keyword dictionary and create feature columns
        for feature_name, keywords in self.keyword_dict.items():
            feature_df[feature_name] = feature_df[self.column].apply(
                lambda text: int(any(re.search(r'\b' + re.escape(keyword) + r'\b', text, flags=re.IGNORECASE) for keyword in keywords))
            )

        # Apply softmax normalization for the job role columns with temperature scaling
        role_columns = list(self.keyword_dict.keys())
        feature_df['Other'] = (feature_df[role_columns].sum(axis=1) == 0).astype(int)

        role_columns = role_columns + ['Other']

        # Softmax with temperature scaling
        def apply_softmax_with_temp(row):
            return softmax(np.array(row) / self.temp)

        feature_df[role_columns] = feature_df[role_columns].apply(apply_softmax_with_temp, axis=1, result_type='expand')

        return feature_df