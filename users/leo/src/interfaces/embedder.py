from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def generate_embeddings():
        pass
