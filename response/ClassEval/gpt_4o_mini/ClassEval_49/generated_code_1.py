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
        This function is used to publish positions, and add the position information to the job_listings list.
        """
        job = {
            'job_title': job_title,
            'company': company,
            'requirements': requirements
        }
        self.job_listings.append(job)

    def remove_job(self, job):
        """
        This function is used to remove positions, and remove the position information from the job_listings list.
        """
        if job in self.job_listings:
            self.job_listings.remove(job)

    def submit_resume(self, name, skills, experience):
        """
        This function is used to submit resumes, and add the resume information to the resumes list.
        """
        resume = {
            'name': name,
            'skills': skills,
            'experience': experience
        }
        self.resumes.append(resume)

    def withdraw_resume(self, resume):
        """
        This function is used to withdraw resumes, and remove the resume information from the resumes list.
        """
        if resume in self.resumes:
            self.resumes.remove(resume)

    def search_jobs(self, criteria):
        """
        This function is used to search for positions, and return the position information that meets the requirements.
        """
        matching_jobs = [
            job for job in self.job_listings if criteria in job['requirements']
        ]
        return matching_jobs

    def get_job_applicants(self, job):
        """
        This function is used to obtain candidate information, and return the candidate information that meets the requirements.
        """
        applicants = [
            resume for resume in self.resumes if any(skill in resume['skills'] for skill in job['requirements'])
        ]
        return applicants


# Test cases
if __name__ == "__main__":
    jobMarketplace = JobMarketplace()
    
    # Test post_job
    jobMarketplace.post_job("Software Engineer", "ABC Company", ['Python', 'Django'])
    print(jobMarketplace.job_listings)  # [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['Python', 'Django']}]

    # Test remove_job
    jobMarketplace.remove_job(jobMarketplace.job_listings[0])
    print(jobMarketplace.job_listings)  # []

    # Test submit_resume
    jobMarketplace.submit_resume("Tom", ['Python', 'Java'], "3 years in software development")
    print(jobMarketplace.resumes)  # [{'name': 'Tom', 'skills': ['Python', 'Java'], 'experience': '3 years in software development'}]

    # Test withdraw_resume
    jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
    print(jobMarketplace.resumes)  # []

    # Test search_jobs
    jobMarketplace.post_job("Software Engineer", "ABC Company", ['Python', 'Django'])
    jobMarketplace.post_job("Data Scientist", "XYZ Inc.", ['Python', 'Machine Learning'])
    print(jobMarketplace.search_jobs("Python"))  
    # [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['Python', 'Django']}, 
    #  {'job_title': 'Data Scientist', 'company': 'XYZ Inc.', 'requirements': ['Python', 'Machine Learning']}]

    # Test get_job_applicants
    jobMarketplace.submit_resume("Tom", ['Python', 'Django'], "3 years in web development")
    print(jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0]))
    # [{'name': 'Tom', 'skills': ['Python', 'Django'], 'experience': '3 years in web development'}]