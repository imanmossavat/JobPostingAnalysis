import pandas as pd
from src.entities.embedding_sample import EmbeddingSample
from src.interfaces.repository import Repository


class EmbeddingsDfRepo(Repository):
    def __init__(self, embeddings_df: pd.DataFrame):
        self.embeddings_df = embeddings_df

    def list(self):
        return EmbeddingSample.from_df(self.embeddings_df)
