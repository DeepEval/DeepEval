class JobMarketplace:
    """
    This is a class that provides functionalities to publish positions, remove positions, submit resumes,
    withdraw resumes, search for positions, and obtain candidate information.
    """

    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title, company, requirements):
        """
        This function is used to publish positions,and add the position information to the job_listings list.
        """
        job = {'job_title': job_title, 'company': company, 'requirements': requirements}
        self.job_listings.append(job)

    def remove_job(self, job):
        """
        This function is used to remove positions,and remove the position information from the job_listings list.
        """
        self.job_listings.remove(job)

    def submit_resume(self, name, skills, experience):
        """
        This function is used to submit resumes,and add the resume information to the resumes list.
        """
        resume = {'name': name, 'skills': skills, 'experience': experience}
        self.resumes.append(resume)

    def withdraw_resume(self, resume):
        """
        This function is used to withdraw resumes,and remove the resume information from the resumes list.
        """
        self.resumes.remove(resume)

    def search_jobs(self, criteria):
        """
        This function is used to search for positions,and return the position information that meets the requirements.
        """
        return [job for job in self.job_listings if criteria in job['requirements']]

    def get_job_applicants(self, job):
        """
        This function is used to obtain candidate information,and return the candidate information that meets the requirements.
        """
        return [resume for resume in self.resumes if self.matches_requirements(job, resume)]

    def matches_requirements(self, job, resume):
        """
        Helper function to determine if a resume matches the job requirements.
        """
        return all(req in resume['skills'] for req in job['requirements'])

# Test cases
if __name__ == "__main__":
    # Initialize the marketplace
    jobMarketplace = JobMarketplace()

    # Test post_job
    jobMarketplace.post_job("Software Engineer", "ABC Company", ['requirement1', 'requirement2'])
    print(jobMarketplace.job_listings)  # [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['requirement1', 'requirement2']}]

    # Test remove_job
    jobMarketplace.remove_job(jobMarketplace.job_listings[0])
    print(jobMarketplace.job_listings)  # []

    # Test submit_resume
    jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
    print(jobMarketplace.resumes)  # [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}]

    # Test withdraw_resume
    jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
    print(jobMarketplace.resumes)  # []

    # Prepare data for further tests
    jobMarketplace.post_job("Software Engineer", "ABC Company", ['skill1', 'skill2'])
    jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")

    # Test search_jobs
    print(jobMarketplace.search_jobs("skill1"))  # [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['skill1', 'skill2']}]

    # Test get_job_applicants
    print(jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0]))  # [{'name': 'Tom', 'skills': ['skill1', 'skill2'], 'experience': 'experience'}]