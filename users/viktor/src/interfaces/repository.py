"""
This module defines the Repository interface, which provides an abstract base class
for repository implementations that handle job post data.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from entities.job_post import JobPost


class IRepository(ABC):
    @abstractmethod
    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[JobPost]:
        """
        Retrieve a list of job posts, optionally filtered by specified criteria.

        Args:
            filters: Optional dictionary containing filter criteria

        Returns:
            List of JobPost objects matching the filter criteria
        """
        pass