
class JobPostFilter():
    def __init__(self):
        pass
    
    def get_jobs_matching_filters(self, repo):
        return repo.list()

    