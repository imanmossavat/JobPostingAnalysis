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

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        jobs = []
        for _, row in df.iterrows():
            job = JobPost(
                job_id=row["job_id"],
                title=row["title"],
                description=row["description"],
                company_name=row["company_name"],
                location=row["location"],
                original_listed_time=row["original_listed_time"],
                language=row["language"],
                skills=row["skills"],
                industries=row["industries"],
            )
            jobs.append(job)
        return cls(jobs)
