from src.entities.embedding import Embedding

EMBEDDING = {
    "id": 1,
    "job_id": "1",
    "model_id": 1,
    "vector": [0.1, 0.2, 0.3],
}


def test_embedding_entity_init():
    embedding = Embedding(**EMBEDDING)

    assert embedding.id == EMBEDDING["id"]
    assert embedding.job_id == EMBEDDING["job_id"]
    assert embedding.model_id == EMBEDDING["model_id"]
    assert embedding.vector == EMBEDDING["vector"]


def test_embedding_entity_from_dict():
    embedding = Embedding.from_dict(EMBEDDING)
    assert embedding.id == EMBEDDING["id"]
    assert embedding.job_id == EMBEDDING["job_id"]
    assert embedding.model_id == EMBEDDING["model_id"]
    assert embedding.vector == EMBEDDING["vector"]


def test_embedding_entity_to_dict():
    embedding = Embedding(**EMBEDDING)
    assert embedding.to_dict() == EMBEDDING
