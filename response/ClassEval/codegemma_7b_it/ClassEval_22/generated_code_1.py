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
        for s in self.students:
            if s["name"] == student["name"] and s["major"] == student["major"]:
                return 0
        self.students.append(student)
        return 1

    def register_class(self, student_name, class_name):
        """
        register a class to the student.
        :param student_name: str
        :param class_name: str
        :return a list of class names that the student has registered
        >>> registration_system = ClassRegistrationSystem()
        >>> registration_system.register_class(student_name="John", class_name="CS101")
        >>> registration_system.register_class(student_name="John", class_name="CS102")
        ["CS101", "CS102"]

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
        >>> registration_system = ClassRegistrationSystem()
        >>> student1 = {"name": "John", "major": "Computer Science"}
        >>> registration_system.register_student(student1)
        >>> registration_system.get_students_by_major("Computer Science")
        ["John"]
        """
        return [student["name"] for student in self.students if student["major"] == major]

    def get_all_major(self):
        """
        get all majors in the system
        :return a list of majors
        >>> registration_system = ClassRegistrationSystem()
        >>> registration_system.students = [{"name": "John", "major": "Computer Science"}],
        >>> registration_system.get_all_major(student1)
        ["Computer Science"]
        """
        return list(set([student["major"] for student in self.students]))

    def get_most_popular_class_in_major(self, major):
        """
        get the class with the highest enrollment in the major.
        :return  a string of the most popular class in this major
        >>> registration_system = ClassRegistrationSystem()
        >>> registration_system.students = [{"name": "John", "major": "Computer Science"},
                                             {"name": "Bob", "major": "Computer Science"},
                                             {"name": "Alice", "major": "Computer Science"}]
        >>> registration_system.students_registration_classes = {"John": ["Algorithms", "Data Structures"],
                                            "Bob": ["Operating Systems", "Data Structures", "Algorithms"]}
        >>> registration_system.get_most_popular_class_in_major("Computer Science")
        "Data Structures"
        """
        if major not in self.students_registration_classes:
            return None
        class_counts = {}
        for student, classes in self.students_registration_classes.items():
            if student["major"] == major:
                for c in classes:
                    if c not in class_counts:
                        class_counts[c] = 0
                    class_counts[c] += 1
        if not class_counts:
            return None
        return max(class_counts, key=class_counts.get)

if __name__ == "__main__":
    registration_system = ClassRegistrationSystem()
    student1 = {"name": "John", "major": "Computer Science"}
    student2 = {"name": "Bob", "major": "Computer Science"}
    student3 = {"name": "Alice", "major": "Computer Science"}
    registration_system.register_student(student1)
    registration_system.register_student(student2)
    registration_system.register_student(student3)
    registration_system.register_class(student_name="John", class_name="Algorithms")
    registration_system.register_class(student_name="John", class_name="Data Structures")
    registration_system.register_class(student_name="Bob", class_name="Operating Systems")
    registration_system.register_class(student_name="Bob", class_name="Data Structures")
    registration_system.register_class(student_name="Bob", class_name="Algorithms")
    print(registration_system.get_students_by_major("Computer Science"))
    print(registration_system.get_all_major())
    print(registration_system.get_most_popular_class_in_major("Computer Science"))