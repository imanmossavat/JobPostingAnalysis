"""Module for managing collections of embeddings with DataFrame conversion capabilities.

This module provides the EmbeddingSample class which handles collections of embeddings
and offers functionality to convert between embedding objects and pandas DataFrames
for easier data manipulation and storage.
"""

import pandas as pd
from src.entities.embedding import Embedding
from typing import List


class EmbeddingSample:
    """A collection of Embedding objects that can be converted to and from a DataFrame.

    This class manages a list of embeddings and provides methods to convert them to
    and from pandas DataFrames for data manipulation and storage.
    """

    def __init__(self, embeddings: List[Embedding]):
        """Initialize an EmbeddingSample with a list of embeddings.

        Args:
            embeddings: List of Embedding objects to be stored
        """
        self.embeddings = embeddings

    def to_df(self) -> pd.DataFrame:
        """Convert the embeddings to a pandas DataFrame.

        Returns:
            DataFrame containing the embeddings with columns: id, job_id, model_id, vector
        """
        return pd.DataFrame(
            [embedding.to_dict() for embedding in self.embeddings],
            columns=["id", "job_id", "model_id", "vector"],
        )

    @classmethod
    def from_df(cls, embedding_df: pd.DataFrame) -> "EmbeddingSample":
        """Create an EmbeddingSample from a pandas DataFrame.

        Args:
            embedding_df: DataFrame containing embedding data with required columns

        Returns:
            A new EmbeddingSample instance containing the embeddings from the DataFrame
        """
        return EmbeddingSample(
            [
                Embedding.from_dict(embedding)
                for embedding in embedding_df.to_dict(orient="records")
            ]
        )
