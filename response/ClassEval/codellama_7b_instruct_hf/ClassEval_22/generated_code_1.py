import collections

class ClassRegistrationSystem:
    def __init__(self):
        self.students = []
        self.students_registration_classes = collections.defaultdict(list)

    def register_student(self, student):
        if student in self.students:
            return 0
        else:
            self.students.append(student)
            self.students_registration_classes[student['name']] = []
            return 1

    def register_class(self, student_name, class_name):
        self.students_registration_classes[student_name].append(class_name)
        return self.students_registration_classes[student_name]

    def get_students_by_major(self, major):
        return [student['name'] for student in self.students if student['major'] == major]

    def get_all_major(self):
        return list(set(student['major'] for student in self.students))

    def get_most_popular_class_in_major(self, major):
        class_counts = collections.Counter(class_name for student in self.students
                                           if student['major'] == major
                                           for class_name in self.students_registration_classes[student['name']])
        return class_counts.most_common(1)[0][0]

if __name__ == "__main__":
    system = ClassRegistrationSystem()

    # Register students
    system.register_student({'name': 'Alice', 'major': 'Computer Science'})
    system.register_student({'name': 'Bob', 'major': 'Mathematics'})
    system.register_student({'name': 'Charlie', 'major': 'Computer Science'})

    # Register classes
    system.register_class('Alice', 'Data Structures')
    system.register_class('Alice', 'Algorithms')
    system.register_class('Bob', 'Calculus')
    system.register_class('Charlie', 'Data Structures')

    # Get students by major
    print(system.get_students_by_major('Computer Science'))  # ['Alice', 'Charlie']

    # Get all majors
    print(system.get_all_major())  # ['Computer Science', 'Mathematics']

    # Get most popular class in major
    print(system.get_most_popular_class_in_major('Computer Science'))  # Data Structures