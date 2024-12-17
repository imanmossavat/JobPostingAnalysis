"""
This module implements the Repository interface using a pandas DataFrame as the storage system.
The DataFrameRepo class provides methods to list and filter job posts stored in a DataFrame.
"""

import pandas as pd
from src.interfaces.repository import Repository
from src.entities.job_post_sample import JobPostSample
from typing import Optional, Dict, Any, List
from src.entities.job_post import JobPost


class DataFrameRepo(Repository):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[JobPost]:
        """
        Retrieve a list of job posts from the DataFrame, optionally filtered by specified criteria.

        Args:
            filters: Optional dictionary that may contain the following keys:
                - 'industries': List of industries to filter by
                - 'skills': List of skills to filter by
                - 'include_companies': List of company names to include

        Returns:
            List of JobPost objects matching the filter criteria

        Example:
            >>> repo.list({'industries': ['IT'], 'skills': ['Python']})
        """
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
