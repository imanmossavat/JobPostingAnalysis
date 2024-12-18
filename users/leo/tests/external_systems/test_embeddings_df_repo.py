import pandas as pd
import pytest
from src.external_systems.embeddings_df_repo import EmbeddingsDfRepo
from src.entities.embedding import Embedding

EMBEDDING_1 = {
    "id": 1,
    "job_id": "1",
    "model_id": 1,
    "vector": [0.1, 0.2, 0.3],
}
EMBEDDING_2 = {
    "id": 2,
    "job_id": "2",
    "model_id": 1,
    "vector": [0.4, 0.5, 0.6],
}
EMBEDDING_3 = {
    "id": 3,
    "job_id": "2",
    "model_id": 2,
    "vector": [0.7, 0.8, 0.9],
}


@pytest.fixture
def embeddings_df():
    return pd.DataFrame([EMBEDDING_1, EMBEDDING_2, EMBEDDING_3])


def test_repository_list_without_parameters(embeddings_df):
    repo = EmbeddingsDfRepo(embeddings_df)
    embeddings_expected = [
        Embedding.from_dict(EMBEDDING_1),
        Embedding.from_dict(EMBEDDING_2),
        Embedding.from_dict(EMBEDDING_3),
    ]
    embeddings_actual = repo.list()
    assert len(embeddings_actual.embeddings) == len(embeddings_expected)
    for i in range(len(embeddings_actual.embeddings)):
        assert embeddings_actual.embeddings[i].id == embeddings_expected[i].id
        assert embeddings_actual.embeddings[i].job_id == embeddings_expected[i].job_id
        assert (
            embeddings_actual.embeddings[i].model_id == embeddings_expected[i].model_id
        )
        assert embeddings_actual.embeddings[i].vector == embeddings_expected[i].vector


def test_repository_list_with_filters(embeddings_df):
    repo = EmbeddingsDfRepo(embeddings_df)
    filters = {"model_id": 1}
    job_ids = set(embeddings_df["job_id"].unique())
    embeddings_expected = [
        Embedding.from_dict(EMBEDDING_1),
        Embedding.from_dict(EMBEDDING_2),
    ]
    embeddings_actual = repo.list(filters, job_ids)
    assert len(embeddings_actual.embeddings) == len(embeddings_expected)
    for i in range(len(embeddings_actual.embeddings)):
        assert embeddings_actual.embeddings[i].id == embeddings_expected[i].id
        assert embeddings_actual.embeddings[i].job_id == embeddings_expected[i].job_id
        assert (
            embeddings_actual.embeddings[i].model_id == embeddings_expected[i].model_id
        )
        assert embeddings_actual.embeddings[i].vector == embeddings_expected[i].vector
