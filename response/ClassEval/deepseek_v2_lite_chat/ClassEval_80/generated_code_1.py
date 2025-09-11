import sqlite3

class SQLQueryBuilder:
    """
    This class provides to build SQL queries, including SELECT, INSERT, UPDATE, and DELETE statements. 
    """

    @staticmethod
    def select(table, columns='*', where=None):
        """
        Generate the SELECT SQL statement from the given parameters.
        :param table: str, the query table in database.
        :param columns: list of str, ['col1', 'col2'].
        :param where: dict, {key1: value1, key2: value2 ...}. The query condition.
        return query: str, the SQL query statement.
        """
        keys = ', '.join(f'{key} = ?' for key in where.keys())
        values = tuple(where.values()) + (columns,)
        query = f"SELECT {', '.join(columns)} FROM {table} WHERE {keys}"
        return query

    @staticmethod
    def insert(table, data):
        """
        Generate the INSERT SQL statement from the given parameters.
        :param table: str, the table to be inserted in database.
        :param data: dict, the key and value in SQL insert statement
        return query: str, the SQL insert statement.
        """
        columns = ', '.join(f'{key}' for key in data.keys())
        values = tuple(data.values())
        query = f"INSERT INTO {table} ({columns}) VALUES ({', '.join("?" for _ in data)})"
        return query

    @staticmethod
    def delete(table, where=None):
        """
        Generate the DELETE SQL statement from the given parameters.
        :param table: str, the table that will be excuted with DELETE operation in database
        :param where: dict, {key1: value1, key2: value2 ...}. The query condition.
        return query: str, the SQL delete statement.
        """
        keys = ', '.join(f'{key} = ?' for key in where.keys())
        query = f"DELETE FROM {table} WHERE {keys}"
        return query

    @staticmethod
    def update(table, data, where=None):
        """
        Generate the UPDATE SQL statement from the given parameters.
        :param table: str, the table that will be excuted with UPDATE operation in database
        :param data: dict, the key and value in SQL update statement
        :param where: dict, {key1: value1, key2: value2 ...}. The query condition.
        return query: str, the SQL update statement.
        """
        updates = ', '.join(f'{key} = ?' for key in data.keys())
        query = f"UPDATE {table} SET {updates} WHERE {SQLQueryBuilder.where(where)}"
        return query

    @staticmethod
    def where(where):
        """
        Helper method to generate the WHERE clause from a dictionary.
        :param where: dict, the query condition.
        return query: str, the WHERE clause.
        """
        if where:
            keys = ', '.join(f'{key} = ?' for key in where.keys())
            return f"WHERE {keys}"
        else:
            return ''

# Test cases
if __name__ == "__main__":
    # Test case for select
    query = SQLQueryBuilder.select('table1', columns=['col1', 'col2'], where={'age': 15})
    print(query)  # Expected output: "SELECT col1, col2 FROM table1 WHERE age='15'"

    # Test case for insert
    data = {'name': 'Test', 'age': 14}
    query = SQLQueryBuilder.insert('table1', data)
    print(query)  # Expected output: "INSERT INTO table1 (name, age) VALUES ('Test', '14')"

    # Test case for delete
    query = SQLQueryBuilder.delete('table1', {'name': 'Test', 'age': 14})
    print(query)  # Expected output: "DELETE FROM table1 WHERE name='Test' AND age='14'"

    # Test case for update
    data = {'name': 'Test2', 'age': 15}
    query = SQLQueryBuilder.update('table1', data, {'name': 'Test'})
    print(query)  # Expected output: "UPDATE table1 SET name='Test2', age='15' WHERE name='Test'"