import numpy as np
import os
import pandas as pd

from src.external_systems.dataframe_repo import DataFrameRepo
from src.external_systems.ssem_embedder import SSEMEmbedder
from src.external_systems.embeddings_df_repo import EmbeddingsDfRepo
from src.external_systems.models_df_repo import ModelsDfRepo
from src.services.job_post_filter import JobPostFilter
from src.services.semantic_search import semantic_search
from src.requests.search_posts import (
    build_search_posts_request,
    build_semantic_search_request,
)


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
EMBEDDING_TEST = np.load("users/leo/data/test_data/description_embedding.npy")

EMBEDDING_1 = {
    "id": 1,
    "job_id": "1",
    "model_id": 1,
    "vector": EMBEDDING_TEST,
}
EMBEDDING_2 = {
    "id": 2,
    "job_id": "2",
    "model_id": 1,
    "vector": EMBEDDING_TEST,
}
EMBEDDING_3 = {
    "id": 3,
    "job_id": "3",
    "model_id": 1,
    "vector": EMBEDDING_TEST,
}
EMBEDDING_1_2 = {
    "id": 4,
    "job_id": "1",
    "model_id": 2,
    "vector": EMBEDDING_TEST,
}
EMBEDDING_2_2 = {
    "id": 5,
    "job_id": "2",
    "model_id": 2,
    "vector": EMBEDDING_TEST,
}
EMBEDDING_3_2 = {
    "id": 6,
    "job_id": "3",
    "model_id": 2,
    "vector": EMBEDDING_TEST,
}

MODELS = pd.DataFrame({"id": [1, 2], "name": ["all-mpnet-base-v2", "model2"]})


request = build_search_posts_request(
    filters={"keyword_search": {"industries": ["Technology"]}}
)
request_semantic = build_semantic_search_request(
    filters={"text": "description", "model_id": 1, "threshold": 0.7}
)
job_data = pd.DataFrame([JOB_1, JOB_2, JOB_3])
embeddings_data = pd.DataFrame(
    [EMBEDDING_1, EMBEDDING_2, EMBEDDING_3, EMBEDDING_1_2, EMBEDDING_2_2, EMBEDDING_3_2]
)

#### EXTERNAL SYSTEMS ####
repo = DataFrameRepo(job_data)
embeddings_repo = EmbeddingsDfRepo(embeddings_data)
models_repo = ModelsDfRepo(MODELS)
model_name = models_repo.list({"model_id": 1})
embedder = SSEMEmbedder(model_name)

#### SERVICES ####
job_filter = JobPostFilter()

### RESPONSES ###
response = semantic_search(
    repo,
    embeddings_repo=embeddings_repo,
    embedder=embedder,
    models_repo=None,
    request=request_semantic,
)
print([job.to_dict() for job in response.value[0].jobs])
print([response.value[1].embeddings[0].to_dict()])
