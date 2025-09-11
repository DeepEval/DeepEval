import sqlite3
import pandas as pd

class DatabaseProcessor:
    """
    This is a class for processing a database, supporting creating tables, inserting data into the database, searching for data based on name, and deleting data from the database.
    """

    def __init__(self, database_name):
        """
        Initialize database name of database processor
        """
        self.database_name = database_name

    def create_table(self, table_name, key1, key2):
        """
        Create a new table in the database if it doesn't exist.
        Make id (INTEGER) as PRIMARY KEY, make key1 as TEXT, key2 as INTEGER.
        :param table_name: str, the name of the table to create.
        :param key1: str, the name of the first column in the table.
        :param key2: str, the name of the second column in the table.
        """
        with sqlite3.connect(self.database_name) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {key1} TEXT,
                    {key2} INTEGER
                )
            ''')
            conn.commit()

    def insert_into_database(self, table_name, data):
        """
        Insert data into the specified table in the database.
        :param table_name: str, the name of the table to insert data into.
        :param data: list, a list of dictionaries where each dictionary represents a row of data.
        """
        with sqlite3.connect(self.database_name) as conn:
            cursor = conn.cursor()
            for entry in data:
                cursor.execute(f'''
                    INSERT INTO {table_name} (name, age)
                    VALUES (?, ?)
                ''', (entry['name'], entry['age']))
            conn.commit()

    def search_database(self, table_name, name):
        """
        Search the specified table in the database for rows with a matching name.
        :param table_name: str, the name of the table to search.
        :param name: str, the name to search for.
        :return: list, a list of tuples representing the rows with matching name, if any;
                 otherwise, returns None.
        """
        with sqlite3.connect(self.database_name) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT * FROM {table_name} WHERE name = ?
            ''', (name,))
            results = cursor.fetchall()
            return results if results else None

    def delete_from_database(self, table_name, name):
        """
        Delete rows from the specified table in the database with a matching name.
        :param table_name: str, the name of the table to delete rows from.
        :param name: str, the name to match for deletion.
        """
        with sqlite3.connect(self.database_name) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                DELETE FROM {table_name} WHERE name = ?
            ''', (name,))
            conn.commit()

if __name__ == "__main__":
    # Test cases
    db = DatabaseProcessor('test_database.db')

    # Test create_table
    db.create_table('user', 'name', 'age')
    print("Table created successfully.")

    # Test insert_into_database
    db.insert_into_database('user', [
        {'name': 'John', 'age': 25},
        {'name': 'Alice', 'age': 30}
    ])
    print("Data inserted successfully.")

    # Test search_database
    result = db.search_database('user', 'John')
    print("Search result:", result)

    # Test delete_from_database
    db.delete_from_database('user', 'John')
    print("Data deleted successfully.")

    # Verify deletion
    result_after_deletion = db.search_database('user', 'John')
    print("Search result after deletion:", result_after_deletion)