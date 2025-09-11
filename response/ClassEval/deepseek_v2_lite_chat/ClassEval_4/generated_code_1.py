class AssessmentSystem:
    def __init__(self):
        """Initialize the students dict in assessment system."""
        self.students = {}

    def add_student(self, name, grade, major):
        """Add a new student into self.students dict"""
        self.students[name] = {'name': name, 'grade': grade, 'major': major, 'courses': {}}

    def add_course_score(self, name, course, score):
        """Add score of specific course for student in self.students"""
        if name in self.students and course in self.students[name]['courses']:
            self.students[name]['courses'][course] += score
        else:
            self.students[name]['courses'][course] = score

    def get_gpa(self, name):
        """Get average grade of one student."""
        if name in self.students and self.students[name]['courses']:
            total_score = sum(self.students[name]['courses'].values())
            return total_score / len(self.students[name]['courses'])
        else:
            return None

    def get_all_students_with_fail_course(self):
        """Get all students who have any score below 60."""
        fail_students = []
        for student_info in self.students.values():
            if any(score < 60 for score in student_info['courses'].values()):
                fail_students.append(student_info['name'])
        return fail_students

    def get_course_average(self, course):
        """Get the average score of a specific course."""
        average_score = None
        for student_info in self.students.values():
            if course in student_info['courses']:
                if student_info['courses'][course] is not None:
                    if average_score is None:
                        average_score = student_info['courses'][course]
                    else:
                        average_score += student_info['courses'][course]
        if average_score is not None:
            average_score /= len(self.students.values())
        return average_score

    def get_top_student(self):
        """Calculate every student's gpa with get_gpa method, and find the student with highest gpa."""
        top_gpa = None
        top_student = None
        for student_name, student_info in self.students.items():
            gpa = self.get_gpa(student_name)
            if gpa is not None and (top_gpa is None or gpa > top_gpa):
                top_gpa = gpa
                top_student = student_name
        return top_student

# Test cases
if __name__ == "__main__":
    system = AssessmentSystem()
    system.add_student('student 1', 3, 'SE')
    system.add_course_score('student 1', 'math', 94)
    system.add_course_score('student 1', 'Computer Network', 92)
    print(system.get_gpa('student 1'))  # Expected output: 93.0
    print(system.get_all_students_with_fail_course())  # Expected output: ['student 1']
    print(system.get_course_average('math'))  # Expected output: 94.0
    print(system.get_top_student())  # Expected output: 'student 1'