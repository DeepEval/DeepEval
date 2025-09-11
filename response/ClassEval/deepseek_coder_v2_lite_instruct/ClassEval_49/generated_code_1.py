class JobMarketplace:
    """
    This is a class that provides functionalities to publish positions, remove positions, submit resumes, withdraw resumes, search for positions, and obtain candidate information.
    """

    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title, company, requirements):
        """
        This function is used to publish positions,and add the position information to the job_listings list.
        :param job_title: The title of the position,str.
        :param company: The company of the position,str.
        :param requirements: The requirements of the position,list.
        :return: None
        """
        job = {
            'job_title': job_title,
            'company': company,
            'requirements': requirements
        }
        self.job_listings.append(job)

    def remove_job(self, job):
        """
        This function is used to remove positions,and remove the position information from the job_listings list.
        :param job: The position information to be removed,dict.
        :return: None
        """
        self.job_listings.remove(job)

    def submit_resume(self, name, skills, experience):
        """
        This function is used to submit resumes,and add the resume information to the resumes list.
        :param name: The name of the resume,str.
        :param skills: The skills of the resume,list.
        :param experience: The experience of the resume,str.
        :return: None
        """
        resume = {
            'name': name,
            'skills': skills,
            'experience': experience
        }
        self.resumes.append(resume)

    def withdraw_resume(self, resume):
        """
        This function is used to withdraw resumes,and remove the resume information from the resumes list.
        :param resume: The resume information to be removed,dict.
        :return: None
        """
        self.resumes.remove(resume)

    def search_jobs(self, criteria):
        """
        This function is used to search for positions,and return the position information that meets the requirements.
        :param criteria: The requirements of the position,str.
        :return: The position information that meets the requirements,list.
        """
        matching_jobs = []
        for job in self.job_listings:
            if criteria in job['requirements']:
                matching_jobs.append(job)
        return matching_jobs

    def get_job_applicants(self, job):
        """
        This function is used to obtain candidate information,and return the candidate information that meets the requirements by calling the matches_requirements function.
        :param job: The position information,dict.
        :return: The candidate information that meets the requirements,list.
        """
        matching_candidates = []
        for resume in self.resumes:
            if all(skill in resume['skills'] for skill in job['requirements']):
                matching_candidates.append(resume)
        return matching_candidates

if __name__ == "__main__":
    jobMarketplace = JobMarketplace()
    
    # Test cases
    jobMarketplace.post_job("Software Engineer", "ABC Company", ['skill1', 'skill2'])
    jobMarketplace.submit_resume("Tom", ['skill1', 'skill2'], "experience")
    
    # Search for jobs
    print("Searching for jobs with 'skill1':")
    print(jobMarketplace.search_jobs('skill1'))
    
    # Get job applicants
    print("Job applicants for 'Software Engineer' position:")
    print(jobMarketplace.get_job_applicants(jobMarketplace.job_listings[0]))
    
    # Remove a job
    jobMarketplace.remove_job(jobMarketplace.job_listings[0])
    print("Job listings after removal:")
    print(jobMarketplace.job_listings)
    
    # Withdraw a resume
    jobMarketplace.withdraw_resume(jobMarketplace.resumes[0])
    print("Resumes after withdrawal:")
    print(jobMarketplace.resumes)