"""Module providing job post filtering functionality.

This module implements the JobPostFilter service class which handles:
- Keyword-based and semantic-based job post filtering
- Repository interaction for fetching job posts
- Response handling for successful and failed operations
"""

from typing import Optional, Dict, Any
from src.responses import (
    ResponseSuccess,
    ResponseFailure,
    ResponseTypes,
    build_response_from_invalid_request,
)
from src.entities.job_post_sample import JobPostSample
from src.interfaces.repository import Repository
from src.requests.search_posts import PostsSearchValidRequest


class JobPostFilter:
    """Service class for filtering and searching job posts."""

    def __init__(self):
        pass

    def _get_jobs_matching_filters(
        self, repo: Repository, filters: Optional[Dict[str, Any]] = None
    ) -> JobPostSample:
        """Retrieve jobs from repository based on filter criteria.

        Args:
            repo: Repository interface implementation for accessing job posts
            filters: Optional filter criteria to apply to the job search

        Returns:
            JobPostSample containing the filtered job posts
        """
        filtered_posts = repo.list(filters=filters)
        return filtered_posts

    def search_jobs(
        self, repo: Repository, request: PostsSearchValidRequest
    ) -> ResponseSuccess | ResponseFailure:
        """Search for jobs using keyword or semantic-based filtering.

        Args:
            repo: Repository interface implementation for accessing job posts
            request: Validated search request containing filter parameters

        Returns:
            ResponseSuccess with matching jobs if successful, or
            ResponseFailure with error details if the operation fails

        Example:
            >>> filter_service = JobPostFilter()
            >>> response = filter_service.search_jobs(repo, valid_request)
            >>> if bool(response):
            >>>     jobs = response.value  # Process successful response
            >>> else:
            >>>     error = response.message  # Handle error case
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
            return ResponseFailure(
                ResponseTypes.PARAMETERS_ERROR,
                "Invalid request parameters. Please check the request and try again.",
            )
        except Exception as exc:
            return ResponseFailure(ResponseTypes.SYSTEM_ERROR, exc)
