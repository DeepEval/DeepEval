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
        self.create_student_table()

    def create_student_table(self):
        """
        Creates a "students" table in the database if it does not exist already.
        :return: None
        """
        with sqlite3.connect(self.database_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    grade INTEGER NOT NULL
                )
            ''')
            conn.commit()

    def insert_student(self, student_data):
        """
        Inserts a new student into the "students" table.
        :param student_data: dict, a dictionary containing the student's information (name, age, gender, grade).
        :return: None
        """
        with sqlite3.connect(self.database_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO students (name, age, gender, grade)
                VALUES (?, ?, ?, ?)
            ''', (student_data['name'], student_data['age'], student_data['gender'], student_data['grade']))
            conn.commit()

    def search_student_by_name(self, name):
        """
        Searches for a student in the "students" table by their name.
        :param name: str, the name of the student to search for.
        :return: list of tuples, the rows from the "students" table that match the search criteria.
        """
        with sqlite3.connect(self.database_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM students WHERE name = ?
            ''', (name,))
            return cursor.fetchall()

    def delete_student_by_name(self, name):
        """
        Deletes a student from the "students" table by their name.
        :param name: str, the name of the student to delete.
        :return: None
        """
        with sqlite3.connect(self.database_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM students WHERE name = ?
            ''', (name,))
            conn.commit()


# Test cases to validate the methods
if __name__ == "__main__":
    processor = StudentDatabaseProcessor("students.db")

    # Test case for inserting a student
    student_data = {'name': 'John', 'age': 15, 'gender': 'Male', 'grade': 9}
    processor.insert_student(student_data)
    print("Inserted student:", student_data)

    # Test case for searching a student by name
    result = processor.search_student_by_name("John")
    print("Search result for 'John':", result)

    # Test case for deleting a student by name
    processor.delete_student_by_name("John")
    print("Deleted student named 'John'.")

    # Verify deletion
    result_after_deletion = processor.search_student_by_name("John")
    print("Search result for 'John' after deletion:", result_after_deletion)