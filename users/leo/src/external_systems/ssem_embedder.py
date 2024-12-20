"""Module implementing sentence embeddings generation using various transformer models.

This module provides the SSEMEmbedder class that can generate embeddings using either
the Sentence-BERT model or other transformer models from the Hugging Face library.
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from src.interfaces.embedder import Embedder


class SSEMEmbedder(Embedder):
    """Sentence embeddings generator using transformer models.

    This class can use either the Sentence-BERT model 'all-mpnet-base-v2' or
    other transformer models to generate sentence embeddings.
    """

    def __init__(self, model_name: str):
        """Initialize the embedder with a specific model.

        Args:
            model_name: Name of the model to use for generating embeddings
        """
        self.model_name = model_name
        if model_name == "all-mpnet-base-v2":
            self.model = SentenceTransformer("all-mpnet-base-v2")
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

    def _encode(self, sentences: List[str]) -> np.ndarray:
        """Internal method to encode sentences into embeddings.

        Args:
            sentences: List of sentences to encode

        Returns:
            Array of embedding vectors for the input sentences
        """
        if self.model_name == "all-mpnet-base-v2":
            return self.model.encode(sentences)
        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return embeddings

    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for a list of sentences.

        Args:
            sentences: List of sentences to generate embeddings for

        Returns:
            Array of embedding vectors for the input sentences
        """
        return self._encode(sentences)
