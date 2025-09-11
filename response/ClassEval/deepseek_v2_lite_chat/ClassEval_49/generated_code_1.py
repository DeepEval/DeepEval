class JobMarketplace:
    """
    This is a class that provides functionalities to publish positions, remove positions, submit resumes, withdraw resumes, search for positions, and obtain candidate information.
    """

    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title, company, requirements):
        """
        This function is used to publish positions, and add the position information to the job_listings list.
        :param job_title: The title of the position, str.
        :param company: The company of the position, str.
        :param requirements: The requirements of the position, list.
        :return: None
        """
        job_listing = {"job_title": job_title, "company": company, "requirements": requirements}
        self.job_listings.append(job_listing)
        return self.job_listings

    def remove_job(self, job):
        """
        This function is used to remove positions, and remove the position information from the job_listings list.
        :param job: The position information to be removed, dict.
        :return: None
        """
        self.job_listings.remove(job)
        return self.job_listings

    def submit_resume(self, name, skills, experience):
        """
        This function is used to submit resumes, and add the resume information to the resumes list.
        :param name: The name of the resume, str.
        :param skills: The skills of the resume, list.
        :param experience: The experience of the resume, str.
        :return: None
        """
        resume = {"name": name, "skills": skills, "experience": experience}
        self.resumes.append(resume)
        return self.resumes

    def withdraw_resume(self, resume):
        """
        This function is used to withdraw resumes, and remove the resume information from the resumes list.
        :param resume: The resume information to be removed, dict.
        :return: None
        """
        self.resumes.remove(resume)
        return self.resumes

    def search_jobs(self, criteria):
        """
        This function is used to search for positions, and return the position information that meets the requirements.
        :param criteria: The requirements of the position, str.
        :return: The position information that meets the requirements, list.
        """
        matching_jobs = [job for job in self.job_listings if criteria in job['requirements']]
        return matching_jobs

    def get_job_applicants(self, job):
        """
        This function is used to obtain candidate information, and return the candidate information that meets the requirements by calling the matches_requirements function.
        :param job: The position information, dict.
        :return: The candidate information that meets the requirements, list.
        """
        matching_applicants = [resume for resume in self.resumes if all(requirement in resume.get('skills', []) for requirement in job['requirements'])]
        return matching_applicants

# Example usage
if __name__ == "__main__":
    job_marketplace = JobMarketplace()
    job_marketplace.post_job("Software Engineer", "ABC Company", ['Python', 'Django'])
    job_marketplace.post_job("Data Analyst", "XYZ Corp", ['SQL', 'Excel'])
    job_marketplace.submit_resume("John Doe", ['Python'], "2 years experience")
    job_marketplace.submit_resume("Jane Smith", ['SQL'], "3 years experience")
    
    print(job_marketplace.search_jobs("Python"))  # Should print [{'job_title': 'Software Engineer', 'company': 'ABC Company', 'requirements': ['Python', 'Django']}]
    print(job_marketplace.get_job_applicants(job_marketplace.job_listings[0]))  # Should print [{'name': 'John Doe', 'skills': ['Python'], 'experience': '2 years experience'}]