import pandas as pd
from src.entities.job_post import JobPost

class JobPostSample:
    def __init__(self, jobs: list[JobPost]):
        self.jobs = jobs


    def to_df(self):
        return pd.DataFrame(
            [job.to_list() for job in self.jobs],
            columns=[
                "job_id",
                "title",
                "description",
                "company_name",
                "location",
                "original_listed_time",
                "language",
                "skills",
                "industries",
            ],
        )
