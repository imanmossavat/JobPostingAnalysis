"""Module providing semantic search functionality for job postings.

This module implements semantic search capabilities using embeddings to find
similar job postings based on text input. It handles:
- Text-to-embedding conversion
- Similarity calculations
- Threshold-based filtering of results
- Response handling for successful and failed operations
"""

import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.responses import (
    ResponseSuccess,
    ResponseFailure,
    ResponseTypes,
    build_response_from_invalid_request,
)
from src.entities.job_post_sample import JobPostSample
from src.entities.embedding_sample import EmbeddingSample
from src.interfaces.repository import Repository
from src.interfaces.embedder import Embedder
from src.requests.search_posts import PostsSearchValidRequest, PostsSearchInvalidRequest


def semantic_search(
    jobs_repo: Repository,
    embeddings_repo: Repository,
    embedder: Embedder,
    models_repo: Optional[Repository],
    request: PostsSearchValidRequest | PostsSearchInvalidRequest,
) -> ResponseSuccess | ResponseFailure:
    """Perform semantic search on job postings using embeddings.

    This function converts input text to embeddings, calculates similarities
    with existing job posting embeddings, and returns matches above a
    specified threshold.

    Args:
        jobs_repo: Repository for accessing job postings
        embeddings_repo: Repository for accessing pre-computed embeddings
        embedder: Service for generating embeddings from text
        models_repo: Optional repository for model metadata
        request: Validated semantic search request containing search parameters
                and threshold

    Returns:
        ResponseSuccess containing tuple of (JobPostSample, EmbeddingSample) if successful,
        ResponseFailure if an error occurs during processing

    Example:
        >>> result = semantic_search(jobs_repo, emb_repo, embedder, models_repo, valid_request)
        >>> if bool(result):
        >>>     jobs, embeddings = result.value
    """
    if not request:  # filtering request, nothing is a valid request

        return build_response_from_invalid_request(request)

    try:
        if not request.filters:  # nothing is a valid request
            jobs, embeddings = jobs_repo.list(filters=None), embeddings_repo.list(
                filters=None
            )
            return ResponseSuccess((jobs, embeddings))

        jobs = jobs_repo.list(filters=None)
        jobs_embeddings = embeddings_repo.list(
            filters=request.filters, job_ids={job.job_id for job in jobs.jobs}
        )
        user_text_embedding = embedder.generate_embeddings([request.filters["text"]])
        # get similar jobs
        data_df = jobs.to_df()
        embeddings_df = jobs_embeddings.to_df()

        similarities = cosine_similarity(
            user_text_embedding, np.vstack(embeddings_df["vector"].to_numpy())
        ).reshape(-1)

        data_df["similarity"] = similarities
        embeddings_df["similarity"] = similarities

        # get jobs with similarity above or equal threshold
        jobs_from_search = data_df.loc[
            data_df["similarity"] >= request.filters["threshold"]
        ]
        embeddings_from_search = embeddings_df.loc[
            embeddings_df["similarity"] >= request.filters["threshold"]
        ]

        return ResponseSuccess(
            (
                JobPostSample.from_df(jobs_from_search.drop(columns=["similarity"])),
                EmbeddingSample.from_df(
                    embeddings_from_search.drop(columns=["similarity"])
                ),
            )
        )

    except Exception as exc:
        return ResponseFailure(ResponseTypes.SYSTEM_ERROR, exc)
