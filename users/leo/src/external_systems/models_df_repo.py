"""Module providing DataFrame-based repository implementation for model metadata.

This module implements the Repository interface for storing and retrieving
model information using pandas DataFrames as the storage mechanism.
"""

import pandas as pd
from typing import Dict
from src.interfaces.repository import Repository


class ModelsDfRepo(Repository):
    """Repository implementation that uses pandas DataFrames to store model information.

    This class provides methods to query model metadata, particularly model names by their IDs.
    """

    def __init__(self, models_df: pd.DataFrame):
        """Initialize the repository with a DataFrame containing model data.

        Args:
            models_df: DataFrame containing the model metadata
        """
        self.models_df = models_df

    def list(self, filters: Dict) -> str:
        """Retrieve a model name based on its ID.

        Args:
            filters: Dictionary containing model_id to filter by

        Returns:
            The name of the model matching the given ID
        """
        return self.models_df.loc[
            self.models_df["id"] == filters.get("model_id"), "name"
        ][0]
