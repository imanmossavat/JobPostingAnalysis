import pandas as pd
import pytest

from src.entities.job_post import JobPost
from src.external_systems.dataframe_repo import DataFrameRepo


JOB_POST_1 = {
    "job_id": "1",
    "title": "title1",
    "description": "description1",
    "company_name": "company1",
    "location": "location1",
    "original_listed_time": 1,
    "language": "english",
    "skills": "Python, Java, C++",
    "industries": "Technology, Software",
}
JOB_POST_2 = {
    "job_id": "2",
    "title": "title2",
    "description": "description2",
    "company_name": "company2",
    "location": "location2",
    "original_listed_time": 2,
    "language": "english",
    "skills": "Java, C++",
    "industries": "Medicine, Software",
}


@pytest.fixture
def jobs_df():
    return pd.DataFrame([JOB_POST_1, JOB_POST_2])


def test_repository_list_without_parameters(jobs_df):
    repo = DataFrameRepo(jobs_df)
    jobs_expected = [JobPost.from_dict(JOB_POST_1), JobPost.from_dict(JOB_POST_2)]
    jobs_actual = repo.list()
    assert len(jobs_actual) == len(jobs_expected)
    for i in range(len(jobs_actual)):
        assert jobs_actual[i].job_id == jobs_expected[i].job_id
        assert jobs_actual[i].title == jobs_expected[i].title
        assert jobs_actual[i].description == jobs_expected[i].description
        assert jobs_actual[i].company_name == jobs_expected[i].company_name
        assert jobs_actual[i].location == jobs_expected[i].location
        assert (
            jobs_actual[i].original_listed_time == jobs_expected[i].original_listed_time
        )
        assert jobs_actual[i].language == jobs_expected[i].language
        assert jobs_actual[i].skills == jobs_expected[i].skills
        assert jobs_actual[i].industries == jobs_expected[i].industries


def test_repository_list_with_industries_in_filter(jobs_df):
    repo = DataFrameRepo(jobs_df)
    filters = {"industries": ["Technology"]}
    jobs_searched = repo.list(filters)
    assert len(jobs_searched) == 1
    assert jobs_searched[0].industries == "Technology, Software"


def test_repository_list_with_skills_in_filter(jobs_df):
    repo = DataFrameRepo(jobs_df)
    filters = {"skills": ["Python"]}
    jobs_searched = repo.list(filters)
    assert len(jobs_searched) == 1
    assert jobs_searched[0].skills == "Python, Java, C++"


def test_repository_list_with_include_company_filter(jobs_df):
    repo = DataFrameRepo(jobs_df)
    filters = {"skills": ["Python"], "include_companies": ["company2"]}
    jobs_searched = repo.list(filters)
    assert len(jobs_searched) == 2
    assert jobs_searched[0].skills == "Python, Java, C++"
    assert jobs_searched[0].company_name == "company1"
    assert jobs_searched[1].company_name == "company2"
    assert jobs_searched[1].skills == "Java, C++"
