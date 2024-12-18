import pandas as pd
from src.entities.embedding_sample import EmbeddingSample
from src.entities.embedding import Embedding

EMBEDDING = {
    "id": 1,
    "job_id": "1",
    "model_id": 1,
    "vector": [0.1, 0.2, 0.3],
}


def test_embedding_sample_to_df():
    embedding = Embedding.from_dict(EMBEDDING)
    embedding_sample = EmbeddingSample([embedding])
    embedding_df = embedding_sample.to_df()
    assert embedding_df["id"][0] == EMBEDDING["id"]
    assert embedding_df["job_id"][0] == EMBEDDING["job_id"]


def test_embedding_sample_from_df():
    embedding_df = pd.DataFrame([EMBEDDING])
    embeddings = EmbeddingSample.from_df(embedding_df)
    assert embeddings.embeddings[0].id == EMBEDDING["id"]
    assert embeddings.embeddings[0].job_id == EMBEDDING["job_id"]
    assert embeddings.embeddings[0].model_id == EMBEDDING["model_id"]
    assert embeddings.embeddings[0].vector == EMBEDDING["vector"]
