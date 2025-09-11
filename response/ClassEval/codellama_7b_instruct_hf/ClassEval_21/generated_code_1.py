from datetime import datetime

class Classroom:
    def __init__(self, id):
        self.id = id
        self.courses = []

    def add_course(self, course):
        if course not in self.courses:
            self.courses.append(course)

    def remove_course(self, course):
        if course in self.courses:
            self.courses.remove(course)

    def is_free_at(self, check_time):
        check_time = datetime.strptime(check_time, '%H:%M')
        for course in self.courses:
            if course['start_time'] <= check_time < course['end_time']:
                return False
        return True

    def check_course_conflict(self, new_course):
        for course in self.courses:
            if course['start_time'] == new_course['start_time'] or course['end_time'] == new_course['end_time']:
                return False
        return True

if __name__ == "__main__":
    classroom = Classroom(1)
    classroom.add_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'})
    print(classroom.is_free_at('10:00'))
    print(classroom.is_free_at('9:00'))
    print(classroom.check_course_conflict({'name': 'SE', 'start_time': '9:40', 'end_time': '10:40'}))
    print(classroom.check_course_conflict({'name': 'SE', 'start_time': '10:00', 'end_time': '11:00'}))