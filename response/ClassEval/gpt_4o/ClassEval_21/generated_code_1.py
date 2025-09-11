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
        Add course to self.courses list if the course wasn't in it and there's no conflict.
        :param course: dict, information of the course, including 'start_time', 'end_time' and 'name'
        """
        if self.check_course_conflict(course):
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
        Check if the classroom is free at the given time.
        :param check_time: str, the time need to be checked in '%H:%M' format
        :return: True if the check_time does not conflict with any course time, or False otherwise.
        """
        check_time = datetime.strptime(check_time, '%H:%M')
        for course in self.courses:
            start_time = datetime.strptime(course['start_time'], '%H:%M')
            end_time = datetime.strptime(course['end_time'], '%H:%M')
            if start_time <= check_time < end_time:
                return False
        return True

    def check_course_conflict(self, new_course):
        """
        Before adding a new course, check if the new course time conflicts with any other course.
        :param new_course: dict, information of the course, including 'start_time', 'end_time' and 'name'
        :return: False if the new course time conflicts with other courses, or True otherwise.
        """
        new_start = datetime.strptime(new_course['start_time'], '%H:%M')
        new_end = datetime.strptime(new_course['end_time'], '%H:%M')
        for course in self.courses:
            start_time = datetime.strptime(course['start_time'], '%H:%M')
            end_time = datetime.strptime(course['end_time'], '%H:%M')
            # Check if there's any overlap
            if (start_time < new_end and new_start < end_time):
                return False
        return True


if __name__ == "__main__":
    # Test cases for each method
    
    # Testing add_course and check_course_conflict
    classroom = Classroom(1)
    classroom.add_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'})
    print(classroom.courses)  # [{'name': 'math', 'start_time': '8:00', 'end_time': '9:40'}]
    
    # Trying to add a conflicting course
    classroom.add_course({'name': 'physics', 'start_time': '9:00', 'end_time': '10:00'})
    print(classroom.courses)  # No change, still only math

    # Testing remove_course
    classroom.remove_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'})
    print(classroom.courses)  # []

    # Testing is_free_at
    classroom.add_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'})
    print(classroom.is_free_at('10:00'))  # True
    print(classroom.is_free_at('9:00'))   # False

    # Testing check_course_conflict directly
    print(classroom.check_course_conflict({'name': 'SE', 'start_time': '9:40', 'end_time': '10:40'}))  # False
    print(classroom.check_course_conflict({'name': 'SE', 'start_time': '10:00', 'end_time': '11:00'})) # True