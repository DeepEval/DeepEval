class AssessmentSystem:
    """
    This is a class for a student assessment system, which supports adding students, adding course scores,
    calculating GPA, and other functions for students and courses.
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
        self.students[name] = {
            'name': name,
            'grade': grade,
            'major': major,
            'courses': {}
        }

    def add_course_score(self, name, course, score):
        """
        Add score of a specific course for student in self.students
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
        :return: average grade (float) if student has courses, or None otherwise
        """
        if name in self.students:
            scores = self.students[name]['courses'].values()
            if scores:
                return sum(scores) / len(scores)
        return None

    def get_all_students_with_fail_course(self):
        """
        Get all students who have any score below 60
        :return: list of str, student names
        """
        return [name for name, info in self.students.items() if any(score < 60 for score in info['courses'].values())]

    def get_course_average(self, course):
        """
        Get the average score of a specific course.
        :param course: str, course name
        :return: float, average scores of this course if anyone has a score, or None if nobody has records.
        """
        total_score = 0
        count = 0
        for student in self.students.values():
            if course in student['courses']:
                total_score += student['courses'][course]
                count += 1
        return total_score / count if count > 0 else None

    def get_top_student(self):
        """
        Calculate every student's GPA with get_gpa method, and find the student with the highest GPA
        :return: str, name of student whose GPA is highest
        """
        top_student = None
        highest_gpa = -1
        for name in self.students:
            gpa = self.get_gpa(name)
            if gpa is not None and gpa > highest_gpa:
                highest_gpa = gpa
                top_student = name
        return top_student


# Test cases for the AssessmentSystem
if __name__ == "__main__":
    system = AssessmentSystem()
    
    # Test add_student
    system.add_student('student 1', 3, 'SE')
    print(system.students)

    # Test add_course_score
    system.add_course_score('student 1', 'math', 94)
    print(system.students)

    # Test get_gpa
    system.add_course_score('student 1', 'Computer Network', 92)
    print(system.get_gpa('student 1'))  # Should print 93.0

    # Test get_all_students_with_fail_course
    system.add_student('student 2', 2, 'SE')
    system.add_course_score('student 1', 'Society', 59)
    system.add_course_score('student 2', 'Society', 65)
    print(system.get_all_students_with_fail_course())  # Should print ['student 1']

    # Test get_course_average
    system.add_course_score('student 1', 'math', 94)
    system.add_course_score('student 2', 'math', 97)
    print(system.get_course_average('math'))  # Should print 95.5

    # Test get_top_student
    system.add_course_score('student 2', 'Computer Network', 98)
    print(system.get_top_student())  # Should print 'student 2'