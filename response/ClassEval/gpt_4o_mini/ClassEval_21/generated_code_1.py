from datetime import datetime

class Classroom:
    """
    This is a class representing a classroom, capable of adding and removing courses,
    checking availability at a given time, and detecting conflicts when scheduling new courses.
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
        Add course to self.courses list if the course wasn't already in it.
        :param course: dict, information of the course, including 'start_time', 'end_time' and 'name'
        """
        if course not in self.courses and not self.check_course_conflict(course):
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
        Check if the time is free in the classroom.
        :param check_time: str, the time need to be checked
        :return: True if the check_time does not conflict with any course time, or False otherwise.
        """
        check_time = datetime.strptime(check_time, '%H:%M').time()
        for course in self.courses:
            start_time = datetime.strptime(course['start_time'], '%H:%M').time()
            end_time = datetime.strptime(course['end_time'], '%H:%M').time()
            if start_time <= check_time <= end_time:
                return False
        return True

    def check_course_conflict(self, new_course):
        """
        Before adding a new course, check if the new course time conflicts with any other course.
        :param new_course: dict, information of the course, including 'start_time', 'end_time' and 'name'
        :return: False if the new course time conflicts (including two courses have the same boundary time)
                 with other courses, or True otherwise.
        """
        new_start_time = datetime.strptime(new_course['start_time'], '%H:%M').time()
        new_end_time = datetime.strptime(new_course['end_time'], '%H:%M').time()
        
        for course in self.courses:
            start_time = datetime.strptime(course['start_time'], '%H:%M').time()
            end_time = datetime.strptime(course['end_time'], '%H:%M').time()
            # Check if there is a conflict
            if not (new_end_time <= start_time or new_start_time >= end_time):
                return False
        return True

# Test cases for each method
if __name__ == "__main__":
    classroom = Classroom(1)
    
    # Test add_course
    course1 = {'name': 'math', 'start_time': '8:00', 'end_time': '9:40'}
    classroom.add_course(course1)
    print(classroom.courses)  # Expected: [{'name': 'math', 'start_time': '8:00', 'end_time': '9:40'}]

    # Test check_course_conflict
    course2 = {'name': 'science', 'start_time': '9:40', 'end_time': '10:40'}
    print(classroom.check_course_conflict(course2))  # Expected: False (no conflict)
    
    course3 = {'name': 'history', 'start_time': '9:30', 'end_time': '10:30'}
    print(classroom.check_course_conflict(course3))  # Expected: False (conflict)
    
    # Test is_free_at
    print(classroom.is_free_at('10:00'))  # Expected: True
    print(classroom.is_free_at('9:00'))   # Expected: False
    
    # Test remove_course
    classroom.remove_course(course1)
    print(classroom.courses)  # Expected: []