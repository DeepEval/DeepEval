import collections

class AssessmentSystem:
    def __init__(self):
        self.students = collections.defaultdict(dict)

    def add_student(self, name, grade, major):
        self.students[name] = {
            'name': name,
            'grade': grade,
            'major': major,
            'courses': {},
        }

    def add_course_score(self, name, course, score):
        if name not in self.students:
            raise ValueError(f"Student {name} not found")
        self.students[name]['courses'][course] = score

    def get_gpa(self, name):
        if name not in self.students:
            raise ValueError(f"Student {name} not found")
        courses = self.students[name]['courses']
        total_score = 0
        total_credits = 0
        for course, score in courses.items():
            total_score += score
            total_credits += 1
        if total_credits == 0:
            return None
        return total_score / total_credits

    def get_all_students_with_fail_course(self):
        return [name for name, student in self.students.items()
                if any(score < 60 for score in student['courses'].values())]

    def get_course_average(self, course):
        scores = [score for score in self.students.values() if course in score['courses']]
        if len(scores) == 0:
            return None
        return sum(scores) / len(scores)

    def get_top_student(self):
        top_student = None
        top_gpa = None
        for name, student in self.students.items():
            gpa = self.get_gpa(name)
            if top_gpa is None or gpa > top_gpa:
                top_gpa = gpa
                top_student = name
        return top_student

if __name__ == "__main__":
    system = AssessmentSystem()
    system.add_student("student 1", 3, "SE")
    system.add_student("student 2", 2, "SE")
    system.add_course_score("student 1", "Computer Network", 92)
    system.add_course_score("student 2", "Computer Network", 97)

    print(system.get_gpa("student 1")) # Should be 93.0
    print(system.get_gpa("student 2")) # Should be 97.0
    print(system.get_all_students_with_fail_course()) # Should be ['student 1']
    print(system.get_course_average("Computer Network")) # Should be 94.5
    print(system.get_top_student()) # Should be 'student 2'