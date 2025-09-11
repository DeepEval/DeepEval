class AssessmentSystem:
    """
    This is a class as an student assessment system, which supports add student, add course score, calculate GPA, and other functions for students and courses.
    """

    def __init__(self):
        """
        Initialize the students dict in assessment system.
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
        Add score of specific course for student in self.students
        :param name: str, student name
        :param course: str, course name
        :param score: int, course score
        """
        if name in self.students:
            self.students[name]['courses'][course] = score

    def get_gpa(self, name):
        """
        Get average grade of one student.
        :param name: str, student name
        :return: if name is in students and this students have courses grade, return average grade(float)
                    or None otherwise
        """
        if name in self.students and self.students[name]['courses']:
            total_score = sum(self.students[name]['courses'].values())
            num_courses = len(self.students[name]['courses'])
            return total_score / num_courses
        return None

    def get_all_students_with_fail_course(self):
        """
        Get all students who have any score below 60
        :return: list of str ,student name
        """
        fail_students = []
        for student in self.students:
            if any(score < 60 for score in self.students[student]['courses'].values()):
                fail_students.append(student)
        return fail_students

    def get_course_average(self, course):
        """
        Get the average score of a specific course.
        :param course: str, course name
        :return: float, average scores of this course if anyone have score of this course, or None if nobody have records.
        """
        scores = [student['courses'].get(course) for student in self.students.values() if course in student['courses']]
        if scores:
            return sum(scores) / len(scores)
        return None

    def get_top_student(self):
        """
        Calculate every student's gpa with get_gpa method, and find the student with highest gpa
        :return: str, name of student whose gpa is highest
        """
        if not self.students:
            return None
        top_student = max(self.students.values(), key=lambda student: self.get_gpa(student['name']))
        return top_student['name']

# Test cases
if __name__ == "__main__":
    system = AssessmentSystem()
    # Test add_student
    system.add_student('student 1', 3, 'SE')
    system.add_student('student 2', 2, 'SE')
    print(system.students)  # Expected: {'student 1': {'name': 'student 1', 'grade': 3, 'major': 'SE', 'courses': {}}, 'student 2': {'name': 'student 2', 'grade': 2, 'major': 'SE', 'courses': {}}}
    
    # Test add_course_score
    system.add_course_score('student 1', 'math', 94)
    system.add_course_score('student 1', 'Computer Network', 92)
    system.add_course_score('student 2', 'Computer Network', 97)
    print(system.students)  # Expected: {'student 1': {'name': 'student 1', 'grade': 3, 'major': 'SE', 'courses': {'math': 94, 'Computer Network': 92}}, 'student 2': {'name': 'student 2', 'grade': 2, 'major': 'SE', 'courses': {'Computer Network': 97}}}
    
    # Test get_gpa
    print(system.get_gpa('student 1'))  # Expected: 93.0
    print(system.get_gpa('student 2'))  # Expected: 97.0
    
    # Test get_all_students_with_fail_course
    system.add_course_score('student 1', 'Society', 59)
    print(system.get_all_students_with_fail_course())  # Expected: ['student 1']
    
    # Test get_course_average
    print(system.get_course_average('Computer Network'))  # Expected: 94.5
    
    # Test get_top_student
    print(system.get_top_student())  # Expected: 'student 2'