"""
This module provides validation and request building functionality for job post searches.
It includes:
- PostsSearchInvalidRequest: For handling and accumulating validation errors
- PostsSearchValidRequest: For representing valid search requests
- build_search_posts_request: Builder function that validates and constructs search requests
"""

from collections.abc import Mapping
from typing import Optional, Dict, Union, Any


class PostsSearchInvalidRequest:
    def __init__(self):
        self.errors = []

    def add_error(self, parameter, message):
        self.errors.append({"parameter": parameter, "message": message})

    def has_errors(self):
        return len(self.errors) > 0

    def __bool__(self):
        return False


class PostsSearchValidRequest:
    def __init__(self, filters):
        self.filters = filters

    def __bool__(self):
        return True


def build_search_posts_request(
    filters: Optional[Dict[str, Any]] = None
) -> Union[PostsSearchValidRequest, PostsSearchInvalidRequest]:
    """
    Build and validate a search request for job posts.

    Args:
        filters: Optional dictionary containing search filters.
                Accepted keys are 'keyword_search' and 'semantic_search'.

    Returns:
        PostsSearchValidRequest if validation passes, or
        PostsSearchInvalidRequest if validation fails with errors.

    Example:
        >>> request = build_search_posts_request({'keyword_search': 'python'})
        >>> if request:
        >>>     # handle valid request
        >>> else:
        >>>     # handle invalid request errors
    """
    accepted_filters = ["keyword_search", "semantic_search"]
    invalid_request = PostsSearchInvalidRequest()

    if filters is not None:
        if not isinstance(filters, Mapping):
            invalid_request.add_error("filters", "Is not iterable")
            return invalid_request

        for key in filters.keys():
            if key not in accepted_filters:
                invalid_request.add_error(
                    "filters", "Key {} cannot be used".format(key)
                )
        if invalid_request.has_errors():
            return invalid_request

    return PostsSearchValidRequest(filters=filters)
