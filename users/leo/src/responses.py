"""
This module provides response handling utilities for the application.
It includes:
- ResponseTypes: Enum-like class defining possible response types
- ResponseFailure: Class for handling error responses
- ResponseSuccess: Class for handling successful responses
- build_response_from_invalid_request: Utility function to create error responses from invalid requests
"""

from src.requests.search_posts import PostsSearchInvalidRequest


class ResponseTypes:
    PARAMETERS_ERROR = "ParametersError"
    RESOURCE_ERROR = "ResourceError"
    SYSTEM_ERROR = "SystemError"
    SUCCESS = "Success"


class ResponseFailure:
    def __init__(self, type_, message):
        self.type = type_
        self.message = self._format_message(message)

    def _format_message(self, msg):
        if isinstance(msg, Exception):
            return f"{msg.__class__.__name__}: {msg}"
        return msg

    @property
    def value(self):
        return {"type": self.type, "message": self.message}

    def __bool__(self):
        return False


class ResponseSuccess:
    def __init__(self, value=None):
        self.type = ResponseTypes.SUCCESS
        self.value = value

    def __bool__(self):
        return True


def build_response_from_invalid_request(
    invalid_request: PostsSearchInvalidRequest,
) -> ResponseFailure:
    """
    Creates a ResponseFailure object from an invalid request.

    Args:
        invalid_request: A PostsSearchInvalidRequest object containing validation errors

    Returns:
        ResponseFailure with PARAMETERS_ERROR type and formatted error messages

    Example:
        >>> invalid_req = PostsSearchInvalidRequest()
        >>> invalid_req.add_error("filters", "Invalid filter")
        >>> response = build_response_from_invalid_request(invalid_req)
    """
    message = "\n".join(
        [f"{err["parameter"]}: {err["message"]}" for err in invalid_request.errors]
    )
    return ResponseFailure(ResponseTypes.PARAMETERS_ERROR, message)
