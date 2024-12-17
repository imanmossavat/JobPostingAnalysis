import pandas as pd
from src.interfaces.repository import Repository
from src.entities.job_post_sample import JobPostSample


class DataFrameRepo(Repository):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def list(self, filters=None):
        if not filters:
            return JobPostSample.from_df(self.data).jobs
        jobs = self.data
        if "industries" in filters:
            selected_industries = filters["industries"]
            jobs = jobs[
                jobs["industries"].apply(
                    lambda x: any(industry in x for industry in selected_industries)
                )
            ]
        if "skills" in filters:
            selected_skills = filters["skills"]
            jobs = jobs[
                jobs["skills"].apply(
                    lambda x: any(skill in x for skill in selected_skills)
                )
            ]
        if "include_companies" in filters:
            selected_companies = filters["include_companies"]
            jobs = pd.concat(
                [jobs, self.data[self.data["company_name"].isin(selected_companies)]]
            ).drop_duplicates(subset="job_id", keep="first")

        return JobPostSample.from_df(jobs).jobs
