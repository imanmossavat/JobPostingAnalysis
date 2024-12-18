import numpy as np
import pandas as pd
import pytest
from unittest import mock
from src.services.semantic_search import semantic_search
from src.entities.job_post import JobPost
from src.entities.job_post_sample import JobPostSample
from src.entities.embedding import Embedding
from src.entities.embedding_sample import EmbeddingSample
from src.requests.search_posts import build_semantic_search_request

JOB_1 = {
    "job_id": "1",
    "title": "title1",
    "description": "description1",
    "company_name": "company1",
    "location": "location1",
    "original_listed_time": 1,
    "language": "english",
    "skills": "Python, Java, C++",
    "industries": "Technology, Software",
}
JOB_2 = {
    "job_id": "2",
    "title": "title2",
    "description": "description2",
    "company_name": "company2",
    "location": "location2",
    "original_listed_time": 2,
    "language": "english",
    "skills": "Java, C++",
    "industries": "Medicine, Software",
}
JOB_3 = {
    "job_id": "3",
    "title": "title3",
    "description": "description3",
    "company_name": "company3",
    "location": "location3",
    "original_listed_time": 3,
    "language": "english",
    "skills": "Python, Java, C++",
    "industries": "Medicine, Software",
}
EMBEDDING_1 = {
    "id": 1,
    "job_id": "1",
    "model_id": 1,
    "vector": np.array([1, 2, 3]),
}
EMBEDDING_2 = {
    "id": 2,
    "job_id": "2",
    "model_id": 1,
    "vector": np.array([1.0001, 2.0002, 3.0003]),
}
EMBEDDING_3 = {
    "id": 3,
    "job_id": "3",
    "model_id": 1,
    "vector": np.array([1.001, 2.002, 3.003]),
}
EMBEDDING_1_2 = {
    "id": 4,
    "job_id": "1",
    "model_id": 2,
    "vector": np.array([1, 2, 3]),
}
EMBEDDING_2_2 = {
    "id": 5,
    "job_id": "2",
    "model_id": 2,
    "vector": np.array([2, 3, 4]),
}
EMBEDDING_3_2 = {
    "id": 6,
    "job_id": "3",
    "model_id": 2,
    "vector": np.array([3, 4, 5]),
}

MODELS = pd.DataFrame({"id": [1, 2], "name": ["model1", "model2"]})


@pytest.fixture
def jobs_sample():
    return JobPostSample(
        [JobPost.from_dict(JOB_1), JobPost.from_dict(JOB_2), JobPost.from_dict(JOB_3)]
    )


@pytest.fixture
def embeddings_sample():
    return EmbeddingSample(
        [
            Embedding.from_dict(EMBEDDING_1),
            Embedding.from_dict(EMBEDDING_2),
            Embedding.from_dict(EMBEDDING_3),
            Embedding.from_dict(EMBEDDING_1_2),
            Embedding.from_dict(EMBEDDING_2_2),
            Embedding.from_dict(EMBEDDING_3_2),
        ]
    )


@pytest.fixture
def models_df():
    return MODELS


def test_semantic_search_without_parameters(jobs_sample, embeddings_sample, models_df):
    jobs_repo = mock.Mock()
    embeddings_repo = mock.Mock()
    embedder = mock.Mock()
    models_repo = mock.Mock()
    jobs_repo.list.return_value = jobs_sample
    embeddings_repo.list.return_value = embeddings_sample
    models_repo.list.return_value = models_df

    request = build_semantic_search_request()
    response = semantic_search(
        jobs_repo, embeddings_repo, embedder, models_repo, request
    )
    assert bool(response) is True
    jobs_repo.list.assert_called_with(filters=None)
    embeddings_repo.list.assert_called_with(filters=None)
    assert response.value == (
        jobs_sample,
        embeddings_sample,
    )


def test_semantic_search_with_filters(jobs_sample, embeddings_sample, models_df):
    jobs_repo = mock.Mock()
    embeddings_repo = mock.Mock()
    jobs_repo.list.return_value = jobs_sample
    embeddings_repo.list.return_value = EmbeddingSample(
        [
            Embedding.from_dict(EMBEDDING_1),
            Embedding.from_dict(EMBEDDING_2),
            Embedding.from_dict(EMBEDDING_3),
        ]
    )
    embedder = mock.Mock()
    embedder.generate_embeddings.return_value = np.array([[1.01, 2.02, 3.03]])
    models_repo = mock.Mock()
    models_repo.list.return_value = models_df
    qry_filters = {"text": "description4", "model_id": 1, "threshold": 0.7}
    request = build_semantic_search_request(filters=qry_filters)
    response = semantic_search(
        jobs_repo, embeddings_repo, embedder, models_repo, request
    )
    assert bool(response) is True
    jobs_repo.list.assert_called_with(filters=None)
    embeddings_repo.list.assert_called_with(
        filters=qry_filters, job_ids={"1", "2", "3"}
    )
    # models_repo.list.assert_called_with(filters=qry_filters)
    assert len(response.value) == 2
    assert response.value[0].to_df().equals(pd.DataFrame([JOB_1, JOB_2, JOB_3]))
    assert (
        response.value[1]
        .to_df()
        .equals(pd.DataFrame([EMBEDDING_1, EMBEDDING_2, EMBEDDING_3]))
    )
