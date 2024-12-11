from abc import ABC, abstractmethod


class Repository(ABC):
    @abstractmethod
    def list(self):
        pass
    