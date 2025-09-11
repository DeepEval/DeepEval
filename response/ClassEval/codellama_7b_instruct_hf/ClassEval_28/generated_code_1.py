import sqlite3
import pandas as pd

class DatabaseProcessor:
    def __init__(self, database_name):
        self.database_name = database_name
        self.connection = sqlite3.connect(self.database_name)
        self.cursor = self.connection.cursor()

    def create_table(self, table_name, key1, key2):
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)")

    def insert_into_database(self, table_name, data):
        self.cursor.executemany(f"INSERT INTO {table_name} VALUES (?, ?)", data)
        self.connection.commit()

    def search_database(self, table_name, name):
        self.cursor.execute(f"SELECT * FROM {table_name} WHERE name = ?", (name,))
        result = self.cursor.fetchall()
        if result:
            return result
        else:
            return None

    def delete_from_database(self, table_name, name):
        self.cursor.execute(f"DELETE FROM {table_name} WHERE name = ?", (name,))
        self.connection.commit()

if __name__ == "__main__":
    db_processor = DatabaseProcessor("example.db")
    db_processor.create_table("test_table", "name", "value")
    
    # Example data to insert
    data_to_insert = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
    db_processor.insert_into_database("test_table", data_to_insert)
    
    # Search for a name
    print(db_processor.search_database("test_table", "Alice"))
    
    # Delete a name
    db_processor.delete_from_database("test_table", "Bob")
    
    # Verify deletion
    print(db_processor.search_database("test_table", "Bob"))
    
    db_processor.connection.close()