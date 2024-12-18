import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModel

from src.interfaces.embedder import Embedder


class SSEMEmbedder(Embedder):
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name == "all-mpnet-base-v2":
            self.model = SentenceTransformer("all-mpnet-base-v2")
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

    def _encode(self, sentences: List[str]) -> np.ndarray:
        if self.model_name == "all-mpnet-base-v2":
            return self.model.encode(sentences)
        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return embeddings

    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        return self._encode(sentences)
