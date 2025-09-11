import sqlite3
import pandas as pd

class DatabaseProcessor:
    """
    This is a class for processing a database, supporting to create tables, insert data into the database, search for data based on name, and delete data from the database.
    """


    def __init__(self, database_name):
        """
        Initialize database name of database processor
        """
        self.database_name = database_name
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name, key1, key2):
        """
        Create a new table in the database if it doesn't exist.
        And make id (INTEGER) as PRIMARY KEY, make key1 as TEXT, key2 as INTEGER
        :param table_name: str, the name of the table to create.
        :param key1: str, the name of the first column in the table.
        :param key2: str, the name of the second column in the table.
        >>> db.create_table('user', 'name', 'age')
        """
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY,
                {key1} TEXT,
                {key2} INTEGER
            )
        """)
        self.conn.commit()

    def insert_into_database(self, table_name, data):
        """
        Insert data into the specified table in the database.
        :param table_name: str, the name of the table to insert data into.
        :param data: list, a list of dictionaries where each dictionary represents a row of data.
        >>> db.insert_into_database('user', [
                {'name': 'John', 'age': 25},
                {'name': 'Alice', 'age': 30}
            ])
        """
        placeholders = ', '.join(['?'] * len(data[0].keys()))
        sql = f"INSERT INTO {table_name} ({', '.join(data[0].keys())}) VALUES ({placeholders})"
        self.cursor.executemany(sql, data)
        self.conn.commit()

    def search_database(self, table_name, name):
        """
        Search the specified table in the database for rows with a matching name.
        :param table_name: str, the name of the table to search.
        :param name: str, the name to search for.
        :return: list, a list of tuples representing the rows with matching name, if any;
                    otherwise, returns None.
        >>> db.search_database('user', 'John')
        [(1, 'John', 25)]
        """
        self.cursor.execute(f"SELECT * FROM {table_name} WHERE {key1} = ?", (name,))
        return self.cursor.fetchall()

    def delete_from_database(self, table_name, name):
        """
        Delete rows from the specified table in the database with a matching name.
        :param table_name: str, the name of the table to delete rows from.
        :param name: str, the name to match for deletion.
        >>> db.delete_from_database('user', 'John')
        """
        self.cursor.execute(f"DELETE FROM {table_name} WHERE {key1} = ?", (name,))
        self.conn.commit()

if __name__ == "__main__":
    db = DatabaseProcessor('mydatabase.db')

    # Test case for create_table
    db.create_table('user', 'name', 'age')
    print("Table created successfully!")

    # Test case for insert_into_database
    data = [
        {'name': 'John', 'age': 25},
        {'name': 'Alice', 'age': 30}
    ]
    db.insert_into_database('user', data)
    print("Data inserted successfully!")

    # Test case for search_database
    result = db.search_database('user', 'John')
    print("Search result:", result)

    # Test case for delete_from_database
    db.delete_from_database('user', 'John')
    print("Data deleted successfully!")