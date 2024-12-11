import pandas as pd
from src.interfaces.repository import Repository
from src.entities.job_post_sample import JobPostSample


class DataFrameRepo(Repository):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def list(self):
        return JobPostSample.from_df(self.data).jobs
