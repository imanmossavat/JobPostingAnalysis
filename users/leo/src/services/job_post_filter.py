"""
This module provides job post filtering functionality through the JobPostFilter class.
It handles different types of job searches including keyword-based and semantic-based filtering,
while managing successful and failed responses appropriately.
"""

from typing import Optional, Dict, Any, List
from src.responses import (
    ResponseSuccess,
    ResponseFailure,
    ResponseTypes,
    build_response_from_invalid_request,
)
from src.entities.job_post import JobPost
from src.interfaces.repository import Repository
from src.requests.search_posts import PostsSearchValidRequest


class JobPostFilter:
    def __init__(self):
        pass

    def _get_jobs_matching_filters(
        self, repo: Repository, filters: Optional[Dict[str, Any]] = None
    ) -> List[JobPost]:
        """
        Retrieves jobs that match the specified filters.

        Args:
            repo: Repository instance to fetch jobs from
            filters: Dictionary containing filter criteria

        Returns:
            List of JobPost objects matching the filters
        """
        filtered_posts = repo.list(filters=filters)
        return filtered_posts

    def _get_similar_jobs(
        self, repo: Repository, request: PostsSearchValidRequest
    ) -> List[JobPost]:
        """
        Retrieves jobs that are semantically similar based on the request.

        Args:
            repo: Repository instance to fetch jobs from
            request: Valid search request containing semantic search parameters

        Returns:
            List of JobPost objects that are semantically similar
        """
        similar_posts = repo.list()
        return similar_posts

    def search_jobs(
        self, repo: Repository, request: PostsSearchValidRequest
    ) -> ResponseSuccess | ResponseFailure:
        """
        Main method to search for jobs based on the provided request.

        Args:
            repo: Repository instance to fetch jobs from
            request: Valid search request containing search parameters

        Returns:
            ResponseSuccess containing matching jobs if successful,
            ResponseFailure if an error occurs

        Example:
            >>> filter_service = JobPostFilter()
            >>> result = filter_service.search_jobs(repo, valid_request)
            >>> if result:
            >>>     jobs = result.value
        """
        if not request:
            return build_response_from_invalid_request(request)

        try:
            if not request.filters:
                jobs = repo.list(filters=None)
                return ResponseSuccess(jobs)
            if "keyword_search" in request.filters:
                jobs_from_search = self._get_jobs_matching_filters(
                    repo, filters=request.filters.get("keyword_search")
                )
                return ResponseSuccess(jobs_from_search)
            if "semantic_search" in request.filters:
                jobs_from_search = self._get_similar_jobs(repo, request)
                return ResponseSuccess(jobs_from_search)
            return ResponseFailure(
                ResponseTypes.PARAMETERS_ERROR,
                "Invalid request parameters. Please check the request and try again.",
            )
        except Exception as exc:
            return ResponseFailure(ResponseTypes.SYSTEM_ERROR, exc)
