class ClassRegistrationSystem:
    """
    This is a class as a class registration system, allowing to register students, register them for classes, retrieve students by major, get a list of all majors, and determine the most popular class within a specific major.
    """

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
        if student['name'] in [s['name'] for s in self.students]:
            return 0
        else:
            self.students.append(student)
            self.students_registration_classes[student['name']] = []
            return 1

    def register_class(self, student_name, class_name):
        """
        register a class to the student.
        :param student_name: str
        :param class_name: str
        :return a list of class names that the student has registered
        """
        if student_name in self.students_registration_classes:
            self.students_registration_classes[student_name].append(class_name)
            return self.students_registration_classes[student_name]
        else:
            return []

    def get_students_by_major(self, major):
        """
        get all students in the major
        :param major: str
        :return a list of student name
        """
        return [student['name'] for student in self.students if student['major'] == major]

    def get_all_major(self):
        """
        get all majors in the system
        :return a list of majors
        """
        return list(set(student['major'] for student in self.students))

    def get_most_popular_class_in_major(self, major):
        """
        get the class with the highest enrollment in the major.
        :param major: str
        :return  a string of the most popular class in this major
        """
        class_count = {}
        for student_name, classes in self.students_registration_classes.items():
            if self.students_by_major(major).__contains__(student_name):
                for class_name in classes:
                    if class_name in class_count:
                        class_count[class_name] += 1
                    else:
                        class_count[class_name] = 1
        if not class_count:
            return None
        most_popular_class = max(class_count, key=class_count.get)
        return most_popular_class

    def students_by_major(self, major):
        return [student['name'] for student in self.students if student['major'] == major]


if __name__ == "__main__":
    registration_system = ClassRegistrationSystem()
    
    # Test register_student
    print(registration_system.register_student({"name": "John", "major": "Computer Science"}))  # Should return 1
    print(registration_system.register_student({"name": "John", "major": "Computer Science"}))  # Should return 0
    
    # Test register_class
    print(registration_system.register_class("John", "Algorithms"))  # Should return ["Algorithms"]
    print(registration_system.register_class("John", "Data Structures"))  # Should return ["Algorithms", "Data Structures"]
    
    # Test get_students_by_major
    print(registration_system.get_students_by_major("Computer Science"))  # Should return ["John"]
    
    # Test get_all_major
    registration_system.students = [{"name": "John", "major": "Computer Science"},
                                     {"name": "Bob", "major": "Computer Science"},
                                     {"name": "Alice", "major": "Computer Science"}]
    print(registration_system.get_all_major())  # Should return ["Computer Science"]
    
    # Test get_most_popular_class_in_major
    registration_system.students_registration_classes = {"John": ["Algorithms", "Data Structures"],
                                                         "Bob": ["Operating Systems", "Data Structures", "Algorithms"]}
    print(registration_system.get_most_popular_class_in_major("Computer Science"))  # Should return "Data Structures"