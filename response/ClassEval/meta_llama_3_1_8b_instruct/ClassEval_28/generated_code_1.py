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

    def create_table(self, table_name, key1, key2):
        """
        Create a new table in the database if it doesn't exist.
        And make id (INTEGER) as PRIMARY KEY, make key1 as TEXT, key2 as INTEGER
        :param table_name: str, the name of the table to create.
        :param key1: str, the name of the first column in the table.
        :param key2: str, the name of the second column in the table.
        >>> db.create_table('user', 'name', 'age')
        """
        try:
            cursor = self.conn.cursor()
            query = f"""
                CREATE TABLE IF NOT EXISTS {table_name}
                (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)
            """
            cursor.execute(query)
            self.conn.commit()
            print(f"Table {table_name} created successfully.")
            return f"Table {table_name} created successfully."
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return f"An error occurred: {e}"

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
        try:
            cursor = self.conn.cursor()
            columns = ', '.join(data[0].keys())
            placeholders = ', '.join('?' for _ in data[0])
            query = f"""
                INSERT INTO {table_name} ({columns}) VALUES ({placeholders})
            """
            values = []
            for row in data:
                values.append(tuple(row.values()))
            cursor.executemany(query, values)
            self.conn.commit()
            print(f"Data inserted successfully into {table_name}.")
            return f"Data inserted successfully into {table_name}."
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return f"An error occurred: {e}"

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
        try:
            cursor = self.conn.cursor()
            query = f"""
                SELECT * FROM {table_name} WHERE {table_name}.name =?
            """
            cursor.execute(query, (name,))
            result = cursor.fetchall()
            if result:
                return result
            else:
                return None
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return f"An error occurred: {e}"

    def delete_from_database(self, table_name, name):
        """
        Delete rows from the specified table in the database with a matching name.
        :param table_name: str, the name of the table to delete rows from.
        :param name: str, the name to match for deletion.
        >>> db.delete_from_database('user', 'John')
        """
        try:
            cursor = self.conn.cursor()
            query = f"""
                DELETE FROM {table_name} WHERE {table_name}.name =?
            """
            cursor.execute(query, (name,))
            self.conn.commit()
            print(f"Data deleted successfully from {table_name}.")
            return f"Data deleted successfully from {table_name}."
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return f"An error occurred: {e}"

    def close_connection(self):
        """
        Close the connection to the database.
        """
        self.conn.close()

if __name__ == "__main__":
    db = DatabaseProcessor('example.db')
    # Create table
    db.create_table('user', 'name', 'age')
    # Insert data
    data = [
        {'name': 'John', 'age': 25},
        {'name': 'Alice', 'age': 30}
    ]
    db.insert_into_database('user', data)
    # Search data
    result = db.search_database('user', 'John')
    print(result)
    # Delete data
    db.delete_from_database('user', 'John')
    # Close connection
    db.close_connection()