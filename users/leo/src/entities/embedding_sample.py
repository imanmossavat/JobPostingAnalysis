import pandas as pd
from src.entities.embedding import Embedding


class EmbeddingSample:
    def __init__(self, embeddings: list[Embedding]):
        self.embeddings = embeddings

    def to_df(self):
        return pd.DataFrame(
            [embedding.to_dict() for embedding in self.embeddings],
            columns=["id", "job_id", "model_id", "vector"],
        )

    @classmethod
    def from_df(cls, embedding_df):
        return EmbeddingSample(
            [
                Embedding.from_dict(embedding)
                for embedding in embedding_df.to_dict(orient="records")
            ]
        )
