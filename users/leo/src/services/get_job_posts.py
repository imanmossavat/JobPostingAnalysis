from src.requests.search_posts import build_search_posts_request
from src.responses import ResponseSuccess

class JobPostFilter():
    def __init__(self):
        pass
    
    def _get_jobs_matching_filters(self, repo, request):
        filtered_posts = repo.list()
        return filtered_posts

    def _get_similar_jobs(self, repo, request):
        similar_posts = repo.list()
        return ResponseSuccess(similar_posts)
    
    def search_jobs(self, repo, request):
        if not request:
            return build_search_posts_request(request)
        
        if not request.filters:
            return ResponseSuccess(repo.list(filters=None))
        if "keyword_search" in request.filters:
            jobs_from_search = self._get_similar_jobs(repo, request)
            return ResponseSuccess(jobs_from_search)
        if "semantic_search" in request.filters:
            jobs_from_search = self._get_similar_jobs(repo, request)
            return ResponseSuccess(jobs_from_search)
