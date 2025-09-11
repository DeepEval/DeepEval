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
            return True
        return False

    def remove_course(self, course):
        """
        Remove course from self.courses list if the course was in it.
        :param course: dict, information of the course, including 'start_time', 'end_time' and 'name'
        """
        if course in self.courses:
            self.courses.remove(course)
            return True
        return False

    def is_free_at(self, check_time):
        """
        Change the time format as '%H:%M' and check the time is free or not in the classroom.
        :param check_time: str, the time need to be checked
        :return: True if the check_time does not conflict with every course time, or False otherwise.
        """
        # Parse the check time
        check_time = datetime.strptime(check_time, "%H:%M")
        # Check if the check time is free
        for course in self.courses:
            start_time = datetime.strptime(course['start_time'], "%H:%M")
            end_time = datetime.strptime(course['end_time'], "%H:%M")
            if (start_time <= check_time < end_time) or (start_time < end_time <= check_time):
                return False
        return True

    def check_course_conflict(self, new_course):
        """
        Before adding a new course, check if the new course time conflicts with any other course.
        :param new_course: dict, information of the course, including 'start_time', 'end_time' and 'name'
        :return: False if the new course time conflicts(including two courses have the same boundary time) with other courses, or True otherwise.
        """
        # Sort the courses by start time
        self.courses.sort(key=lambda x: (int(x['start_time'].split(':')[0]), int(x['start_time'].split(':')[1])))

        # Check if the new course conflicts with any existing course
        for i, course in enumerate(self.courses):
            start_time = datetime.strptime(course['start_time'], "%H:%M")
            end_time = datetime.strptime(course['end_time'], "%H:%M")
            if (start_time < new_course['start_time'] and new_course['start_time'] <= end_time) or \
               (start_time < new_course['end_time'] and new_course['end_time'] <= end_time) or \
               (start_time <= new_course['start_time'] and new_course['start_time'] < end_time):
                return False
        return True

# Test cases
if __name__ == "__main__":
    # Test add_course
    classroom = Classroom(1)
    assert classroom.add_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'}) == True
    assert classroom.add_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'}) == False

    # Test remove_course
    classroom.add_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'})
    assert classroom.remove_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'}) == True
    assert classroom.remove_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'}) == False

    # Test is_free_at
    classroom.add_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'})
    classroom.add_course({'name': 'SE', 'start_time': '9:40', 'end_time': '10:40'})
    assert classroom.is_free_at('10:00') == True
    assert classroom.is_free_at('9:00') == False

    # Test check_course_conflict
    classroom.add_course({'name': 'math', 'start_time': '8:00', 'end_time': '9:40'})
    classroom.check_course_conflict({'name': 'SE', 'start_time': '9:40', 'end_time': '10:40'}) == False
    classroom.check_course_conflict({'name': 'SE', 'start_time': '9:00', 'end_time': '10:00'}) == True