class AssessmentSystem:
    """
    This is a class as a student assessment system, which supports adding students, adding course scores, calculating GPA, and other functions for students and courses.
    """

    def __init__(self):
        """
        Initialize the students dict in the assessment system.
        """
        self.students = {}

    def add_student(self, name, grade, major):
        """
        Add a new student into self.students dict
        :param name: str, student name
        :param grade: int, student grade
        :param major: str, student major
        """
        if name not in self.students:
            self.students[name] = {'name': name, 'grade': grade, 'major': major, 'courses': {}}

    def add_course_score(self, name, course, score):
        """
        Add score of a specific course for a student in self.students
        :param name: str, student name
        :param course: str, course name
        :param score: int, course score
        """
        if name in self.students:
            self.students[name]['courses'][course] = score

    def get_gpa(self, name):
        """
        Get the average grade of one student.
        :param name: str, student name
        :return: float, average grade if available, None otherwise
        """
        if name in self.students:
            courses = self.students[name]['courses']
            if courses:
                return sum(courses.values()) / len(courses)
        return None

    def get_all_students_with_fail_course(self):
        """
        Get all students who have any score below 60
        :return: list of str, student names
        """
        failed_students = []
        for student, data in self.students.items():
            if any(score < 60 for score in data['courses'].values()):
                failed_students.append(student)
        return failed_students

    def get_course_average(self, course):
        """
        Get the average score of a specific course.
        :param course: str, course name
        :return: float, average score of this course if records exist, None otherwise
        """
        total_score = 0
        count = 0
        for student_data in self.students.values():
            if course in student_data['courses']:
                total_score += student_data['courses'][course]
                count += 1
        return total_score / count if count > 0 else None

    def get_top_student(self):
        """
        Calculate every student's GPA with get_gpa method, and find the student with the highest GPA
        :return: str, name of the student whose GPA is highest
        """
        top_student = None
        highest_gpa = -1
        for student in self.students:
            gpa = self.get_gpa(student)
            if gpa is not None and gpa > highest_gpa:
                highest_gpa = gpa
                top_student = student
        return top_student


if __name__ == "__main__":
    system = AssessmentSystem()

    # Test case for add_student
    system.add_student('student 1', 3, 'SE')
    print(system.students)  # {'student 1': {'name': 'student 1', 'grade': 3, 'major': 'SE', 'courses': {}}}

    # Test case for add_course_score
    system.add_course_score('student 1', 'math', 94)
    print(system.students)  # {'student 1': {'name': 'student 1', 'grade': 3, 'major': 'SE', 'courses': {'math': 94}}}

    # Test case for get_gpa
    system.add_course_score('student 1', 'Computer Network', 92)
    print(system.get_gpa('student 1'))  # 93.0

    # Test case for get_all_students_with_fail_course
    system.add_course_score('student 1', 'Society', 59)
    print(system.get_all_students_with_fail_course())  # ['student 1']

    # Test case for get_course_average
    system.add_student('student 2', 2, 'SE')
    system.add_course_score('student 2', 'math', 88)
    print(system.get_course_average('math'))  # 91.0

    # Test case for get_top_student
    system.add_course_score('student 2', 'Computer Network', 97)
    print(system.get_top_student())  # 'student 2'