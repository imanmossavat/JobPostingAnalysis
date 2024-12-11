"""
job_post.py

This module defines the JobPost class, which represents a job posting with various attributes such as job ID, title, description, company name, location, original listed time, language, skills, and industries.
"""

class JobPost:
    def __init__(
        self,
        job_id: str,
        title: str,
        description: str,
        company_name: str,
        location: str,
        original_listed_time: int,
        language: str,
        skills: str,
        industries: str,
    ):

        self.job_id = job_id
        self.title = title
        self.description = description
        self.company_name = company_name
        self.location = location
        self.original_listed_time = original_listed_time
        self.language = language
        self.skills = skills
        self.industries = industries

    def to_list(self):
        return [
            self.job_id,
            self.title,
            self.description,
            self.company_name,
            self.location,
            self.original_listed_time,
            self.language,
            self.skills,
            self.industries,
        ]
    
    @classmethod
    def from_dict(cls, job_dict):
        return JobPost(**job_dict)

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "title": self.title,
            "description": self.description,
            "company_name": self.company_name,
            "location": self.location,
            "original_listed_time": self.original_listed_time,
            "language": self.language,
            "skills": self.skills,
            "industries": self.industries,
        }   
        