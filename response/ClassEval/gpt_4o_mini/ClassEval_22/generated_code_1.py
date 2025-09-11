class ClassRegistrationSystem:
    """
    This is a class as a class registration system, allowing to register students, register them for classes, 
    retrieve students by major, get a list of all majors, and determine the most popular class within a specific major.
    """

    def __init__(self):
        """
        Initialize the registration system with the attribute students and students_registration_classes.
        students is a list of student dictionaries, each student dictionary has the keys of name and major.
        students_registration_classes is a dictionary, key is the student name, value is a list of class names.
        """
        self.students = []
        self.students_registration_classes = {}

    def register_student(self, student):
        """
        Register a student to the system. If the student is already registered, return 0, else return 1.
        """
        for s in self.students:
            if s['name'] == student['name']:
                return 0  # Student already registered
        
        self.students.append(student)
        return 1  # Successfully registered

    def register_class(self, student_name, class_name):
        """
        Register a class to the student.
        :param student_name: str
        :param class_name: str
        :return: a list of class names that the student has registered
        """
        if student_name not in self.students_registration_classes:
            self.students_registration_classes[student_name] = []
        
        self.students_registration_classes[student_name].append(class_name)
        return self.students_registration_classes[student_name]

    def get_students_by_major(self, major):
        """
        Get all students in the major.
        :param major: str
        :return: a list of student names
        """
        return [s['name'] for s in self.students if s['major'] == major]

    def get_all_major(self):
        """
        Get all majors in the system.
        :return: a list of majors
        """
        return list(set(s['major'] for s in self.students))

    def get_most_popular_class_in_major(self, major):
        """
        Get the class with the highest enrollment in the major.
        :param major: str
        :return: a string of the most popular class in this major
        """
        class_count = {}
        for student in self.students:
            if student['major'] == major:
                student_classes = self.students_registration_classes.get(student['name'], [])
                for cls in student_classes:
                    if cls not in class_count:
                        class_count[cls] = 0
                    class_count[cls] += 1
        
        # Find the class with the highest enrollment
        if not class_count:
            return None  # No classes found for this major
        
        most_popular_class = max(class_count, key=class_count.get)
        return most_popular_class


if __name__ == "__main__":
    # Test cases
    registration_system = ClassRegistrationSystem()

    # Test register_student
    student1 = {"name": "John", "major": "Computer Science"}
    output = registration_system.register_student(student1)
    print(output)  # Expected: 1 (registered successfully)
    
    output = registration_system.register_student(student1)
    print(output)  # Expected: 0 (already registered)

    # Test register_class
    output = registration_system.register_class(student_name="John", class_name="CS101")
    print(output)  # Expected: ["CS101"]
    
    output = registration_system.register_class(student_name="John", class_name="CS102")
    print(output)  # Expected: ["CS101", "CS102"]

    # Test get_students_by_major
    student2 = {"name": "Alice", "major": "Computer Science"}
    registration_system.register_student(student2)
    output = registration_system.get_students_by_major("Computer Science")
    print(output)  # Expected: ["John", "Alice"]

    # Test get_all_major
    student3 = {"name": "Bob", "major": "Mathematics"}
    registration_system.register_student(student3)
    output = registration_system.get_all_major()
    print(output)  # Expected: ["Computer Science", "Mathematics"]

    # Test get_most_popular_class_in_major
    registration_system.students_registration_classes = {
        "John": ["Algorithms", "Data Structures"],
        "Alice": ["Algorithms", "Operating Systems"],
        "Bob": ["Data Structures", "Algorithms"],
    }
    output = registration_system.get_most_popular_class_in_major("Computer Science")
    print(output)  # Expected: "Algorithms" or "Data Structures" depending on the implementation.