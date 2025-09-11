class ClassRegistrationSystem:
    def __init__(self):
        """
        Initialize the registration system with the attribute students and students_registration_class.
        students is a list of student dictionaries, each student dictionary has the key of name and major.
        students_registration_class is a dictionaries, key is the student name, value is a list of class names
        """
        self.students = []
        self.students_registration_classes = {}

    def register_student(self, student):
        """
        register a student to the system, add the student to the students list, if the student is already registered, return 0, else return 1
        """
        if any(student['name'] == s['name'] and student['major'] == s['major'] for s in self.students):
            return 0
        self.students.append(student)
        return 1

    def register_class(self, student_name, class_name):
        """
        register a class to the student.
        :param student_name: str
        :param class_name: str
        :return a list of class names that the student has registered
        """
        if student_name not in self.students_registration_classes:
            self.students_registration_classes[student_name] = []
        self.students_registration_classes[student_name].append(class_name)
        return self.students_registration_classes[student_name]

    def get_students_by_major(self, major):
        """
        get all students in the major
        :param major: str
        :return a list of student name
        """
        return [student['name'] for student in [s for s in self.students if s['major'] == major]]

    def get_all_major(self):
        """
        get all majors in the system
        :return a list of majors
        """
        return [student['major'] for student in self.students]

    def get_most_popular_class_in_major(self, major):
        """
        get the class with the highest enrollment in the major.
        :return  a string of the most popular class in this major
        """
        student_classes = [self.students_registration_classes[student_name] for student_name in self.students_registration_classes if self.students_registration_classes[student_name] is not None]
        total_classes = set().union(*student_classes)
        class_counts = {class_name: student_classes.count(class_name) for class_name in total_classes}
        most_popular_class = max(class_counts, key=class_counts.get)
        return most_popular_class

# Test cases
def test_class_registration_system():
    registration_system = ClassRegistrationSystem()

    # Test register_student
    assert registration_system.register_student({"name": "John", "major": "Computer Science"}) == 1
    assert registration_system.register_student({"name": "John", "major": "Computer Science"}) == 0

    # Test register_class
    assert registration_system.register_class("John", "Algorithms") == ["Algorithms"]
    assert registration_system.register_class("John", "CS101") == ["Algorithms", "CS101"]

    # Test get_students_by_major
    registration_system.students = [{"name": "John", "major": "Computer Science"}, {"name": "Bob", "major": "Computer Science"}]
    assert registration_system.get_students_by_major("Computer Science") == ["John", "Bob"]

    # Test get_all_major
    registration_system.students = [{"name": "John", "major": "Computer Science"}, {"name": "Bob", "major": "Physics"}, {"name": "Alice", "major": "Computer Science"}]
    assert registration_system.get_all_major() == ["Computer Science", "Physics"]

    # Test get_most_popular_class_in_major
    registration_system.students_registration_classes = {"John": ["Algorithms", "CS101"], "Bob": ["Algorithms", "Operating Systems", "CS101"], "Alice": ["Operating Systems", "CS101"]}
    assert registration_system.get_most_popular_class_in_major("Computer Science") == "CS101"

if __name__ == "__main__":
    test_class_registration_system()
    print("All tests passed!")