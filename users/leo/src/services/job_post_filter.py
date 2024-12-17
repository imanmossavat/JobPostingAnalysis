from src.responses import (
    ResponseSuccess,
    ResponseFailure,
    ResponseTypes,
    build_response_from_invalid_request,
)


class JobPostFilter:
    def __init__(self):
        pass

    def _get_jobs_matching_filters(self, repo, filters=None):
        filtered_posts = repo.list(filters=filters)
        return filtered_posts

    def _get_similar_jobs(self, repo, request):  # to modify
        similar_posts = repo.list()
        return similar_posts

    def search_jobs(self, repo, request):
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
