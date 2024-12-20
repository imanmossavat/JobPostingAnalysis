"""Module defining the abstract interface for embedding generators.

This module provides the Embedder interface that must be implemented by
any class that generates embeddings from job postings.
"""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base class for embedding generation implementations.

    Any class that generates embeddings must implement this interface,
    specifically the generate_embeddings method.
    """

    @abstractmethod
    def generate_embeddings(self):
        """Generate embeddings from input data.

        This method must be implemented by concrete classes to define
        how embeddings are generated from the source data.
        """
        pass
