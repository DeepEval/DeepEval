class JobMarketplace:
    """
    This is a class that provides functionalities to publish positions, remove positions, submit resumes, withdraw resumes, search for positions, and obtain candidate information.
    """

    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title, company, requirements):
        """
        This function is used to publish positions and add the position information to the job_listings list.
        
        :param job_title: The title of the position, str.
        :param company: The company of the position, str.
        :param requirements: The requirements of the position, list.
        :return: None
        """
        new_job = {
            'job_title': job_title,
            'company': company,
           'requirements': requirements
        }
        self.job_listings.append(new_job)
        return new_job

    def remove_job(self, job):
        """
        This function is used to remove positions and remove the position information from the job_listings list.
        
        :param job: The position information to be removed, dict.
        :return: None
        """
        self.job_listings = [j for j in self.job_listings if j!= job]
        return self.job_listings

    def submit_resume(self, name, skills, experience):
        """
        This function is used to submit resumes and add the resume information to the resumes list.
        
        :param name: The name of the resume, str.
        :param skills: The skills of the resume, list.
        :param experience: The experience of the resume, str.
        :return: None
        """
        new_resume = {
            'name': name,
           'skills': skills,
            'experience': experience
        }
        self.resumes.append(new_resume)
        return new_resume

    def withdraw_resume(self, resume):
        """
        This function is used to withdraw resumes and remove the resume information from the resumes list.
        
        :param resume: The resume information to be removed, dict.
        :return: None
        """
        self.resumes = [r for r in self.resumes if r!= resume]
        return self.resumes

    def search_jobs(self, criteria):
        """
        This function is used to search for positions and return the position information that meets the requirements.
        
        :param criteria: The requirements of the position, str.
        :return: The position information that meets the requirements, list.
        """
        matching_jobs = [job for job in self.job_listings if criteria in job['requirements']]
        return matching_jobs

    def matches_requirements(self, job, resume):
        """
        This is a helper function to check if a resume matches the requirements of a job.
        
        :param job: The position information, dict.
        :param resume: The resume information, dict.
        :return: True if the resume matches the requirements, False otherwise.
        """
        return all(req in resume['skills'] for req in job['requirements'])

    def get_job_applicants(self, job):
        """
        This function is used to obtain candidate information and return the candidate information that meets the requirements.
        
        :param job: The position information, dict.
        :return: The candidate information that meets the requirements, list.
        """
        matching_resumes = [resume for resume in self.resumes if self.matches_requirements(job, resume)]
        return matching_resumes

if __name__ == "__main__":
    job_marketplace = JobMarketplace()

    # Test case for post_job
    job_marketplace.post_job("Software Engineer", "ABC Company", ['requirement1','requirement2'])
    print(job_marketplace.job_listings)
    # Output: [{'job_title': 'Software Engineer', 'company': 'ABC Company','requirements': ['requirement1','requirement2']}]

    # Test case for remove_job
    job_marketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['requirement1','requirement2']}]
    job_marketplace.remove_job(job_marketplace.job_listings[0])
    print(job_marketplace.job_listings)
    # Output: []

    # Test case for submit_resume
    job_marketplace.submit_resume("Tom", ['skill1','skill2'], "experience")
    print(job_marketplace.resumes)
    # Output: [{'name': 'Tom','skills': ['skill1','skill2'], 'experience': 'experience'}]

    # Test case for withdraw_resume
    job_marketplace.withdraw_resume(job_marketplace.resumes[0])
    print(job_marketplace.resumes)
    # Output: []

    # Test case for search_jobs
    job_marketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1','skill2']}]
    print(job_marketplace.search_jobs("skill1"))
    # Output: [{'job_title': 'Software Engineer', 'company': 'ABC Company','requirements': ['skill1','skill2']}]

    # Test case for get_job_applicants
    job_marketplace.resumes = [{"name": "Tom", "skills": ['skill1','skill2'], "experience": "experience"}]
    job_marketplace.job_listings = [{"job_title": "Software Engineer", "company": "ABC Company", "requirements": ['skill1','skill2']}]
    print(job_marketplace.get_job_applicants(job_marketplace.job_listings[0]))
    # Output: [{'name': 'Tom','skills': ['skill1','skill2'], 'experience': 'experience'}]