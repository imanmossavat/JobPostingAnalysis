import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.responses import (
    ResponseSuccess,
    ResponseFailure,
    ResponseTypes,
    build_response_from_invalid_request,
)
from src.entities.job_post_sample import JobPostSample
from src.entities.embedding_sample import EmbeddingSample


def semantic_search(repo, embeddings_repo, embedder, models_repo, request):
    if not request:

        return build_response_from_invalid_request(request)

    try:
        if not request.filters:
            jobs, embeddings = repo.list(filters=None), embeddings_repo.list(
                filters=None
            )
            return ResponseSuccess((jobs, embeddings))

        data = repo.list(filters=None)
        embeddings = embeddings_repo.list(
            filters=request.filters, job_ids={job.job_id for job in data.jobs}
        )
        # model = models_repo.list(request.filters)
        input_text_vector = embedder.generate_embeddings([request.filters["text"]])
        # get similar jobs
        data_df = data.to_df()
        embeddings_df = embeddings.to_df()

        similarities = cosine_similarity(
            input_text_vector, np.vstack(embeddings_df["vector"].to_numpy())
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
