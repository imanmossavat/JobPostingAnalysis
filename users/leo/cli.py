import pandas as pd
from src.external_systems.dataframe_repo import DataFrameRepo
from src.services.get_job_posts import JobPostFilter


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
    "skills": "Python, Java, C++",
    "industries": "Technology, Software",
}

data = pd.DataFrame([JOB_POST_1, JOB_POST_2])
repo = DataFrameRepo(data)
job_filter = JobPostFilter()
result = job_filter.get_jobs_matching_filters(repo)
print([job.to_dict() for job in result])
