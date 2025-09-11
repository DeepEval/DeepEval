from datetime import datetime

class Classroom:
    """
    This is a class representing a classroom, capable of adding and removing courses, checking availability at a given time, and detecting conflicts when scheduling new courses.
    """

    def __init__(self, id):
        """
        Initialize the classroom management system.
        :param id: int, the id of classroom
        """
        self.id = id
        self.courses = []

    def add_course(self, course):
        """
        Add course to self.courses list if the course wasn't in it.
        :param course: dict, information of the course, including 'start_time', 'end_time' and 'name'
        """
        if course not in self.courses:
            self.courses.append(course)

    def remove_course(self, course):
        """
        Remove course from self.courses list if the course was in it.
        :param course: dict, information of the course, including 'start_time', 'end_time' and 'name'
        """
        if course in self.courses:
            self.courses.remove(course)

    def is_free_at(self, check_time):
        """
        Change the time format as '%H:%M' and check the time is free or not in the classroom.
        :param check_time: str, the time need to be checked
        :return: True if the check_time does not conflict with every course time, or False otherwise.
        """
        time_format = '%H:%M'
        check_time = datetime.strptime(check_time, time_format).time()
        for course in self.courses:
            start_time = datetime.strptime(course['start_time'], time_format).time()
            end_time = datetime.strptime(course['end_time'], time_format).time()
            if start_time <= check_time < end_time:
                return False
        return True

    def check_course_conflict(self, new_course):
        """
        Before adding a new course, check if the new course time conflicts with any other course.
        :param new_course: dict, information of the course, including 'start_time', 'end_time' and 'name'
        :return: False if the new course time conflicts(including two courses have the same boundary time) with other courses, or True otherwise.
        """
        time_format = '%H:%M'
        new_start_time = datetime.strptime(new_course['start_time'], time_format).time()
        new_end_time = datetime.strptime(new_course['end_time'], time_format).time()
        for course in self.courses:
            start_time = datetime.strptime(course['start_time'], time_format).time()
            end_time = datetime.strptime(course['end_time'], time_format).time()
            if (new_start_time < end_time and new_end_time > start_time) or \
               (new_start_time == end_time and new_end_time == start_time):
                return False
        return True

if __name__ == "__main__":
    classroom = Classroom(1)
    print(classroom.is_free_at('10:00'))  # True
    print(classroom.is_free_at('9:00'))   # False
    classroom.add_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'})
    print(classroom.is_free_at('9:00'))   # False
    print(classroom.is_free_at('10:00'))  # True
    print(classroom.check_course_conflict({'name': 'SE', 'start_time': '9:40', 'end_time': '10:40'}))  # False
    classroom.add_course({'name': 'SE', 'start_time': '9:40', 'end_time': '10:40'})
    print(classroom.check_course_conflict({'name': 'SE', 'start_time': '9:40', 'end_time': '10:40'}))  # False
    print(classroom.check_course_conflict({'name': 'SE', 'start_time': '10:40', 'end_time': '11:40'}))  # True