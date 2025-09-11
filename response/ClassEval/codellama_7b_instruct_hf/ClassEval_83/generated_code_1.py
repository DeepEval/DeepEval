import sqlite3

class StudentDatabaseProcessor:
    def __init__(self, database_name):
        self.database_name = database_name

    def create_student_table(self):
        with sqlite3.connect(self.database_name) as connection:
            cursor = connection.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, gender TEXT, grade INTEGER)")

    def insert_student(self, student_data):
        with sqlite3.connect(self.database_name) as connection:
            cursor = connection.cursor()
            cursor.execute("INSERT INTO students (name, age, gender, grade) VALUES (?, ?, ?, ?)", (student_data["name"], student_data["age"], student_data["gender"], student_data["grade"]))

    def search_student_by_name(self, name):
        with sqlite3.connect(self.database_name) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM students WHERE name = ?", (name,))
            return cursor.fetchall()

    def delete_student_by_name(self, name):
        with sqlite3.connect(self.database_name) as connection:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM students WHERE name = ?", (name,))

    
if __name__ == "__main__":

    # Test cases

    # Test case 1: Create student table
    processor = StudentDatabaseProcessor("students.db")
    processor.create_student_table()

    # Test case 2: Insert student
    student_data = {"name": "John", "age": 15, "gender": "Male", "grade": 9}
    processor.insert_student(student_data)

    # Test case 3: Search student by name
    result = processor.search_student_by_name("John")
    print(result)

    # Test case 4: Delete student by name
    processor.delete_student_by_name("John")