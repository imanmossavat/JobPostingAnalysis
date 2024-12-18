import pytest
import numpy as np

from src.external_systems.ssem_embedder import SSEMEmbedder


def test_generate_embedding_with_sentence_transformer():
    embedder = SSEMEmbedder("all-mpnet-base-v2")
    text = "Hello, world!"
    embedding = embedder.generate_embeddings([text])
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[1] == 768
