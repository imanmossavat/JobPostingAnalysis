import pandas as pd

from src.entities.job_post_sample import JobPostSample
from src.entities.job_post import JobPost

JOB_POST = {
    "job_id": "1",
    "title": "title1",
    "description": "description1",
    "company_name": "company1",
    "location": "location1",
    "original_listed_time": 1625309472,
    "language": "english",
    "skills": "Python, Java, C++",
    "industries": "Technology, Software",
}

def test_job_post_sample_init():
    jobs = [
        JOB_POST,
    ]
    job_posts = [JobPost(**job) for job in jobs]
    job_post_sample = JobPostSample(job_posts)

    job = job_post_sample.jobs[0]
    assert job.job_id == "1"
    assert job.title == "title1"
    assert job.description == "description1"
    assert job.company_name == "company1"
    assert job.location == "location1"
    assert job.original_listed_time == 1625309472
    assert job.language == "english"
    assert job.skills == "Python, Java, C++"
    assert job.industries == "Technology, Software"

def test_job_post_sample_to_df():
    jobs_df = pd.DataFrame([JOB_POST])
    job_post_sample = JobPostSample([JobPost(**JOB_POST)])
    job_posts_df = job_post_sample.to_df()

    assert job_posts_df.equals(jobs_df)