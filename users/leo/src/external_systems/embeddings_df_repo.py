import pandas as pd
from src.entities.embedding_sample import EmbeddingSample
from src.interfaces.repository import Repository


class EmbeddingsDfRepo(Repository):
    def __init__(self, embeddings_df: pd.DataFrame):
        self.embeddings_df = embeddings_df

    def list(self, filters=None, job_ids=None):
        if not filters:
            return EmbeddingSample.from_df(self.embeddings_df)
        embeddings = self.embeddings_df[
            self.embeddings_df["model_id"] == filters["model_id"]
        ]
        return EmbeddingSample.from_df(embeddings[embeddings["job_id"].isin(job_ids)])
