"""Module providing DataFrame-based repository implementation for embeddings.

This module implements the Repository interface for storing and retrieving
embeddings using pandas DataFrames as the storage mechanism.
"""

import pandas as pd
from typing import Optional, Dict, List
from src.entities.embedding_sample import EmbeddingSample
from src.interfaces.repository import Repository


class EmbeddingsDfRepo(Repository):
    """Repository implementation that uses pandas DataFrames to store embeddings.

    This class provides methods to query and filter embeddings stored in a DataFrame.
    """

    def __init__(self, embeddings_df: pd.DataFrame):
        """Initialize the repository with a DataFrame containing embeddings.

        Args:
            embeddings_df: DataFrame containing the embedding data
        """
        self.embeddings_df = embeddings_df

    def list(
        self, filters: Optional[Dict] = None, job_ids: Optional[List[str]] = None
    ) -> EmbeddingSample:
        """Retrieve embeddings based on optional filters and job IDs.

        Args:
            filters: Dictionary of filter criteria (currently supports model_id)
            job_ids: List of job IDs to filter by

        Returns:
            EmbeddingSample containing the filtered embeddings
        """
        if not filters:
            return EmbeddingSample.from_df(self.embeddings_df)
        embeddings = self.embeddings_df[
            self.embeddings_df["model_id"] == filters["model_id"]
        ]
        return EmbeddingSample.from_df(embeddings[embeddings["job_id"].isin(job_ids)])
