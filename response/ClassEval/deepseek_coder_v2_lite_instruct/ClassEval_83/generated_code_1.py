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

    def create_student_table(self):
        """
        Creates a "students" table in the database if it does not exist already.
        Fields include ID of type int, name of type str, age of type int, gender of type str, and grade of type int
        :return: None
        """
        conn = sqlite3.connect(self.database_name)
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
        conn.close()

    def insert_student(self, student_data):
        """
        Inserts a new student into the "students" table.
        :param student_data: dict, a dictionary containing the student's information (name, age, gender, grade).
        :return: None
        """
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO students (name, age, gender, grade)
            VALUES (?, ?, ?, ?)
        ''', (student_data['name'], student_data['age'], student_data['gender'], student_data['grade']))
        conn.commit()
        conn.close()

    def search_student_by_name(self, name):
        """
        Searches for a student in the "students" table by their name.
        :param name: str, the name of the student to search for.
        :return: list of tuples, the rows from the "students" table that match the search criteria.
        """
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM students WHERE name = ?
        ''', (name,))
        result = cursor.fetchall()
        conn.close()
        return result

    def delete_student_by_name(self, name):
        """
        Deletes a student from the "students" table by their name.
        :param name: str, the name of the student to delete.
        :return: None
        """
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM students WHERE name = ?
        ''', (name,))
        conn.commit()
        conn.close()

# Test cases
if __name__ == "__main__":
    processor = StudentDatabaseProcessor("students.db")
    processor.create_student_table()

    # Insert a student
    student_data = {'name': 'John', 'age': 15, 'gender': 'Male', 'grade': 9}
    processor.insert_student(student_data)

    # Search for a student
    result = processor.search_student_by_name("John")
    print("Search Result:", result)

    # Delete a student
    processor.delete_student_by_name("John")

    # Search for the deleted student
    result = processor.search_student_by_name("John")
    print("Search Result after deletion:", result)