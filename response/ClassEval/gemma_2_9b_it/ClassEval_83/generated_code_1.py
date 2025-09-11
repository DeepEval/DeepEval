import sqlite3

class StudentDatabaseProcessor:
    """
    This is a class with database operation, including inserting student information, searching for student information by name, and deleting student information by name.
    """

    def __init__(self, database_name):
        """
        Initializes the StudentDatabaseProcessor object with the specified database name.
        :param database_name: str, the name of the SQLite database.
        """
        self.database_name = database_name
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()

    def create_student_table(self):
        """
        Creates a "students" table in the database if it does not exist already.Fields include ID of type int, name of type str, age of type int, gender of type str, and grade of type int
        :return: None
        >>> processor = StudentDatabaseProcessor("students.db")
        >>> processor.create_student_table()
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                grade INTEGER
            )
        ''')
        self.conn.commit()

    def insert_student(self, student_data):
        """
        Inserts a new student into the "students" table.
        :param student_data: dict, a dictionary containing the student's information (name, age, gender, grade).
        :return: None
        >>> processor = StudentDatabaseProcessor("students.db")
        >>> processor.create_student_table()
        >>> student_data = {'name': 'John', 'age': 15, 'gender': 'Male', 'grade': 9}
        >>> processor.insert_student(student_data)
        """
        name, age, gender, grade = student_data['name'], student_data['age'], student_data['gender'], student_data['grade']
        self.cursor.execute("INSERT INTO students (name, age, gender, grade) VALUES (?, ?, ?, ?)", (name, age, gender, grade))
        self.conn.commit()

    def search_student_by_name(self, name):
        """
        Searches for a student in the "students" table by their name.
        :param name: str, the name of the student to search for.
        :return: list of tuples, the rows from the "students" table that match the search criteria.
        >>> processor = StudentDatabaseProcessor("students.db")
        >>> processor.create_student_table()
        >>> result = processor.search_student_by_name("John")
        """
        self.cursor.execute("SELECT * FROM students WHERE name = ?", (name,))
        return self.cursor.fetchall()

    def delete_student_by_name(self, name):
        """
        Deletes a student from the "students" table by their name.
        :param name: str, the name of the student to delete.
        :return: None
        >>> processor = StudentDatabaseProcessor("students.db")
        >>> processor.create_student_table()
        >>> student_data = {'name': 'John', 'age': 15, 'gender': 'Male', 'grade': 9}
        >>> processor.insert_student(student_data)
        >>> processor.delete_student_by_name("John")
        """
        self.cursor.execute("DELETE FROM students WHERE name = ?", (name,))
        self.conn.commit()

    def __del__(self):
        self.conn.close()

if __name__ == "__main__":
    processor = StudentDatabaseProcessor("students.db")
    processor.create_student_table()

    student_data = {'name': 'John', 'age': 15, 'gender': 'Male', 'grade': 9}
    processor.insert_student(student_data)

    result = processor.search_student_by_name("John")
    print(result)  # Output: [(1, 'John', 15, 'Male', 9)]

    processor.delete_student_by_name("John")

    result = processor.search_student_by_name("John")
    print(result)  # Output: []