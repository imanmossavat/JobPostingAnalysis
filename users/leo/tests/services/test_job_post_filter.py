import pytest
from unittest import mock
from src.services.job_post_filter import JobPostFilter
from src.entities.job_post import JobPost
from src.entities.job_post_sample import JobPostSample
from src.requests.search_posts import build_search_posts_request
from src.responses import ResponseTypes


JOB_3 = JobPost(
    job_id="3",
    title="title3",
    description="description3",
    company_name="company3",
    location="location3",
    original_listed_time=3,
    language="english",
    skills="Python, Java, C++",
    industries="Medicine, Software",
)


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

    return JobPostSample([job_1, job_2, JOB_3])


@pytest.fixture
def entity_only_job_post():
    return JobPostSample([JOB_3])


def test_filter_posts_without_parameters(entity_job_post):
    repo = mock.Mock()
    repo.list.return_value = entity_job_post
    job_filter = JobPostFilter()

    request = build_search_posts_request()
    response = job_filter.search_jobs(repo, request)

    assert bool(response) is True
    repo.list.assert_called_with(filters=None)
    assert response.value == entity_job_post


def test_filter_posts_with_keyword_search(entity_only_job_post):
    repo = mock.Mock()
    repo.list.return_value = entity_only_job_post
    qry_filters = {"keyword_search": {"industries": ["Medicine"]}}
    job_filter = JobPostFilter()
    request = build_search_posts_request(filters=qry_filters)
    response = job_filter.search_jobs(repo, request)

    assert bool(response) is True
    repo.list.assert_called_with(filters=list(qry_filters.values())[0])
    assert response.value == entity_only_job_post


def test_filter_posts_handles_generic_error():
    repo = mock.Mock()
    repo.list.side_effect = Exception("Just an error message")
    job_filter = JobPostFilter()
    request = build_search_posts_request(filters={})
    response = job_filter.search_jobs(repo, request)
    assert bool(response) is False
    assert response.value == {
        "type": ResponseTypes.SYSTEM_ERROR,
        "message": "Exception: Just an error message",
    }


## test_filter_posts_with_semantic_search MISSING


def test_filter_posts_handles_bad_request():
    repo = mock.Mock()
    job_filter = JobPostFilter()
    request = build_search_posts_request(filters=5)
    response = job_filter.search_jobs(repo, request)
    assert bool(response) is False
    assert response.value == {
        "type": ResponseTypes.PARAMETERS_ERROR,
        "message": "filters: Is not iterable",
    }
