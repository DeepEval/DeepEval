class ClassRegistrationSystem:
    """
    This is a class as a class registration system, allowing to register students, register them for classes, 
    retrieve students by major, get a list of all majors, and determine the most popular class within a specific major.
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
        for existing_student in self.students:
            if existing_student["name"] == student["name"]:
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
        students_in_major = [student["name"] for student in self.students if student["major"] == major]
        return students_in_major

    def get_all_major(self):
        """
        get all majors in the system
        :return a list of majors
        """
        majors = [student["major"] for student in self.students]
        return list(set(majors))

    def get_most_popular_class_in_major(self, major):
        """
        get the class with the highest enrollment in the major.
        :return  a string of the most popular class in this major
        """
        # Get all students in the major
        students_in_major = [student for student in self.students if student["major"] == major]
        
        # Get all classes registered by students in the major
        all_classes = [class_name for student in students_in_major for class_name in self.students_registration_classes.get(student["name"], [])]
        
        # Count the occurrence of each class
        class_counts = {}
        for class_name in all_classes:
            if class_name not in class_counts:
                class_counts[class_name] = 1
            else:
                class_counts[class_name] += 1
        
        # Find the class with the highest count
        most_popular_class = max(class_counts, key=class_counts.get)
        
        return most_popular_class


if __name__ == "__main__":
    registration_system = ClassRegistrationSystem()

    student1 = {"name": "John", "major": "Computer Science"}
    output = registration_system.register_student(student1)
    print("Output for register_student:", output)
    student2 = {"name": "John", "major": "Computer Science"}
    output = registration_system.register_student(student2)
    print("Output for register_student:", output)

    registration_system = ClassRegistrationSystem()
    output = registration_system.register_class("John", "CS101")
    print("Output for register_class:", output)
    output = registration_system.register_class("John", "CS102")
    print("Output for register_class:", output)

    registration_system = ClassRegistrationSystem()
    student1 = {"name": "John", "major": "Computer Science"}
    student2 = {"name": "Alice", "major": "Computer Science"}
    student3 = {"name": "Bob", "major": "Mathematics"}
    registration_system.register_student(student1)
    registration_system.register_student(student2)
    registration_system.register_student(student3)
    output = registration_system.get_students_by_major("Computer Science")
    print("Output for get_students_by_major:", output)

    registration_system = ClassRegistrationSystem()
    student1 = {"name": "John", "major": "Computer Science"}
    student2 = {"name": "Alice", "major": "Computer Science"}
    student3 = {"name": "Bob", "major": "Mathematics"}
    registration_system.register_student(student1)
    registration_system.register_student(student2)
    registration_system.register_student(student3)
    output = registration_system.get_all_major()
    print("Output for get_all_major:", output)

    registration_system = ClassRegistrationSystem()
    student1 = {"name": "John", "major": "Computer Science"}
    student2 = {"name": "Bob", "major": "Computer Science"}
    student3 = {"name": "Alice", "major": "Computer Science"}
    registration_system.register_student(student1)
    registration_system.register_student(student2)
    registration_system.register_student(student3)
    registration_system.students_registration_classes = {
        "John": ["Algorithms", "Data Structures"],
        "Bob": ["Operating Systems", "Data Structures", "Algorithms"],
        "Alice": ["Algorithms", "Data Structures"]
    }
    output = registration_system.get_most_popular_class_in_major("Computer Science")
    print("Output for get_most_popular_class_in_major:", output)