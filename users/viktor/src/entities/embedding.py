from typing import List


class Embedding:
    def __init__(self, id: int, job_id: str, model_id: int, vector: List[float]):
        self.id = id
        self.job_id = job_id
        self.model_id = model_id
        self.vector = vector

    @classmethod
    def from_dict(cls, embedding_dict):
        return Embedding(**embedding_dict)

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "model_id": self.model_id,
            "vector": self.vector,
        }
