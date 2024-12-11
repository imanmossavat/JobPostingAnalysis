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

def test_job_entity_init():
    job = JobPost(**JOB_POST)

    assert job.job_id == JOB_POST["job_id"]
    assert job.title == JOB_POST["title"]
    assert job.description == JOB_POST["description"]
    assert job.company_name == JOB_POST["company_name"]
    assert job.location == JOB_POST["location"]
    assert job.original_listed_time == JOB_POST["original_listed_time"]
    assert job.language == JOB_POST["language"]
    assert job.skills == JOB_POST["skills"]
    assert job.industries == JOB_POST["industries"]

def test_job_entity_to_list():
    job_attributes_expected = [
        JOB_POST["job_id"],
        JOB_POST["title"],
        JOB_POST["description"],
        JOB_POST["company_name"],
        JOB_POST["location"],
        JOB_POST["original_listed_time"],
        JOB_POST["language"],
        JOB_POST["skills"],
        JOB_POST["industries"],
    ]
    job = JobPost(**JOB_POST)
    job_list = job.to_list()

    assert job_list == job_attributes_expected


def test_job_entity_from_dict():
    job = JobPost.from_dict(JOB_POST)
    assert job.job_id == JOB_POST["job_id"]
    assert job.title == JOB_POST["title"]
    assert job.description == JOB_POST["description"]
    assert job.company_name == JOB_POST["company_name"]
    assert job.location == JOB_POST["location"]
    assert job.original_listed_time == JOB_POST["original_listed_time"]
    assert job.language == JOB_POST["language"]
    assert job.skills == JOB_POST["skills"]
    assert job.industries == JOB_POST["industries"]
