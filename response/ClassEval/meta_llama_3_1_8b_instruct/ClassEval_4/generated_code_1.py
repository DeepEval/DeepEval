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
        >>> system.add_student('student 1', 3, 'SE')
        >>> system.students
        {'student 1': {'name':'student 1', 'grade': 3,'major': 'SE', 'courses': {}}}
        """
        self.students[name] = {'name': name, 'grade': grade,'major': major, 'courses': {}}

    def add_course_score(self, name, course, score):
        """
        Add score of specific course for student in self.students
        :param name: str, student name
        :param course: str, cource name
        :param score: int, cource score
        >>> system.add_student('student 1', 3, 'SE')
        >>> system.add_course_score('student 1','math', 94)
        >>> system.students
        {'student 1': {'name':'student 1', 'grade': 3,'major': 'SE', 'courses': {'math': 94}}}
        """
        if name in self.students:
            self.students[name]['courses'][course] = score
        else:
            print(f"Student {name} does not exist.")

    def get_gpa(self, name):
        """
        Get average grade of one student.
        :param name: str, student name
        :return: if name is in students and this students have courses grade, return average grade(float)
                    or None otherwise
        >>> system.add_student('student 1', 3, 'SE')
        >>> system.add_course_score('student 1','math', 94)
        >>> system.add_course_score('student 1', 'Computer Network', 92)
        >>> system.get_gpa('student 1')
        93.0
        """
        if name in self.students and self.students[name]['courses']:
            total_score = sum(self.students[name]['courses'].values())
            num_courses = len(self.students[name]['courses'])
            return total_score / num_courses
        else:
            return None

    def get_all_students_with_fail_course(self):
        """
        Get all students who have any score blow 60
        :return: list of str,student name
        >>> system.add_course_score('student 1', 'Society', 59)
        >>> system.get_all_students_with_fail_course()
        ['student 1']
        """
        fail_students = []
        for student in self.students.values():
            if any(score < 60 for score in student['courses'].values()):
                fail_students.append(student['name'])
        return fail_students

    def get_course_average(self, course):
        """
        Get the average score of a specific course.
        :param course: str, course name
        :return: float, average scores of this course if anyone have score of this course, or None if nobody have records.
        """
        total_score = 0
        count = 0
        for student in self.students.values():
            if course in student['courses']:
                total_score += student['courses'][course]
                count += 1
        if count > 0:
            return total_score / count
        else:
            return None

    def get_top_student(self):
        """
        Calculate every student's gpa with get_gpa method, and find the student with highest gpa
        :return: str, name of student whose gpa is highest
        >>> system.add_student('student 1', 3, 'SE')
        >>> system.add_student('student 2', 2, 'SE')
        >>> system.add_course_score('student 1', 'Computer Network', 92)
        >>> system.add_course_score('student 2', 'Computer Network', 97)
        >>> system.get_top_student()
       'student 2'
        """
        top_student = max(self.students, key=lambda student: self.get_gpa(student))
        return top_student


if __name__ == "__main__":
    system = AssessmentSystem()

    # Test case for add_student method
    system.add_student('student 1', 3, 'SE')
    system.add_student('student 2', 2, 'SE')
    print(system.students)
    # Expected output:
    # {'student 1': {'name':'student 1', 'grade': 3,'major': 'SE', 'courses': {}}, 
    # 'student 2': {'name':'student 2', 'grade': 2,'major': 'SE', 'courses': {}}}

    # Test case for add_course_score method
    system.add_course_score('student 1','math', 94)
    print(system.students)
    # Expected output:
    # {'student 1': {'name':'student 1', 'grade': 3,'major': 'SE', 'courses': {'math': 94}}, 
    # 'student 2': {'name':'student 2', 'grade': 2,'major': 'SE', 'courses': {}}}

    # Test case for get_gpa method
    print(system.get_gpa('student 1'))  # Should print 94.0
    print(system.get_gpa('student 2'))  # Should print None

    # Test case for get_all_students_with_fail_course method
    system.add_course_score('student 2', 'Society', 59)
    print(system.get_all_students_with_fail_course())  # Should print ['student 2']

    # Test case for get_course_average method
    print(system.get_course_average('math'))  # Should print 94.0

    # Test case for get_top_student method
    print(system.get_top_student())  # Should print'student 2'