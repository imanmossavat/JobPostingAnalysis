import pytest
from unittest import mock
from src.services.get_job_posts import JobPostFilter
from src.entities.job_post import JobPost
from src.entities.job_post_sample import JobPostSample


@pytest.fixture
def entity_job_post():
    job_1 = JobPost(
        job_id="1",
        title="title1",
        description="description1",
        company_name="company1",
        location="location1",
        original_listed_time=1,
        language="english",
        skills="Python, Java, C++",
        industries="Technology, Software",
    )
    job_2 = JobPost(
        job_id="2",
        title="title2",
        description="description2",
        company_name="company2",
        location="location2",
        original_listed_time=2,
        language="english",
        skills="Python, Java, C++",
        industries="Technology, Software",
    )
    job_3 = JobPost(
        job_id="3",
        title="title3",
        description="description3",
        company_name="company3",
        location="location3",
        original_listed_time=3,
        language="english",
        skills="Python, Java, C++",
        industries="Technology, Software",
    )
    return JobPostSample([job_1, job_2, job_3])
    

def test_filter_posts_without_parameters(entity_job_post):
    repo = mock.Mock()
    repo.list.return_value = entity_job_post
    job_filter = JobPostFilter()
    result = job_filter.get_jobs_matching_filters(repo)
    repo.list.assert_called_with()
    assert result == entity_job_post


