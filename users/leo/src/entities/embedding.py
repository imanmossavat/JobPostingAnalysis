"""Module containing the Embedding class for representing vector embeddings of job postings.

This module defines the structure and methods for handling individual embedding instances,
including their creation, conversion to and from dictionaries.
"""

from typing import List, Dict, Any


class Embedding:
    """Represents a vector embedding for a job posting.

    This class stores the embedding vector along with its metadata including
    the unique identifier, job identifier, and the model used to create it.
    """

    def __init__(self, id: int, job_id: str, model_id: int, vector: List[float]):
        """Initialize an Embedding instance.

        Args:
            id: Unique identifier for the embedding
            job_id: Identifier of the associated job posting
            model_id: Identifier of the model used to create the embedding
            vector: List of floating point numbers representing the embedding vector
        """
        self.id = id
        self.job_id = job_id
        self.model_id = model_id
        self.vector = vector

    @classmethod
    def from_dict(cls, embedding_dict: Dict[str, Any]) -> "Embedding":
        """Create an Embedding instance from a dictionary.

        Args:
            embedding_dict: Dictionary containing the embedding data with required fields

        Returns:
            A new Embedding instance with the data from the dictionary
        """
        return Embedding(**embedding_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Embedding instance to a dictionary.

        Returns:
            Dictionary containing the embedding data with keys: id, job_id, model_id, vector
        """
        return {
            "id": self.id,
            "job_id": self.job_id,
            "model_id": self.model_id,
            "vector": self.vector,
        }
