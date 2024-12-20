"""
This module defines the Repository interface, which provides an abstract base class
for repository implementations that handle job post data.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from src.entities.job_post_sample import JobPostSample


class Repository(ABC):
    @abstractmethod
    def list(self, filters: Optional[Dict[str, Any]] = None) -> JobPostSample:
        """
        Retrieve a list of job posts, optionally filtered by specified criteria.

        Args:
            filters: Optional dictionary containing filter criteria

        Returns:
            A JobPostSample object, holding JobPost objects matching the filter criteria
        """
        pass
