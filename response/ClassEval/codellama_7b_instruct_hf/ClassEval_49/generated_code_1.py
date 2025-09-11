import json

class JobMarketplace:
    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title, company, requirements):
        new_job = {
            "job_title": job_title,
            "company": company,
            "requirements": requirements
        }
        self.job_listings.append(new_job)

    def remove_job(self, job):
        self.job_listings.remove(job)

    def submit_resume(self, name, skills, experience):
        new_resume = {
            "name": name,
            "skills": skills,
            "experience": experience
        }
        self.resumes.append(new_resume)

    def withdraw_resume(self, resume):
        self.resumes.remove(resume)

    def search_jobs(self, criteria):
        matching_jobs = []
        for job in self.job_listings:
            if criteria in job["requirements"]:
                matching_jobs.append(job)
        return matching_jobs

    def get_job_applicants(self, job):
        matching_resumes = []
        for resume in self.resumes:
            if self.matches_requirements(resume, job):
                matching_resumes.append(resume)
        return matching_resumes

    def matches_requirements(self, resume, job):
        for requirement in job["requirements"]:
            if requirement not in resume["skills"]:
                return False
        return True

if __name__ == "__main__":
    job_marketplace = JobMarketplace()
    job_marketplace.post_job("Software Engineer", "ABC Company", ["skill1", "skill2"])
    job_marketplace.submit_resume("Tom", ["skill1", "skill3"], "experience")
    job_marketplace.submit_resume("Jerry", ["skill2", "skill4"], "experience")
    print(job_marketplace.search_jobs("skill1"))
    print(job_marketplace.get_job_applicants(job_marketplace.job_listings[0]))