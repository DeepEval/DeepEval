class SQLQueryBuilder:
    """
    This class provides to build SQL queries, including SELECT, INSERT, UPDATE, and DELETE statements. 
    """

    @staticmethod
    def select(table, columns='*', where=None):
        """
        Generate the SELECT SQL statement from the given parameters.
        """
        if isinstance(columns, list):
            columns = ', '.join(columns)
        query = f"SELECT {columns} FROM {table}"
        if where:
            conditions = ' AND '.join([f"{key}='{value}'" for key, value in where.items()])
            query += f" WHERE {conditions}"
        return query

    @staticmethod
    def insert(table, data):
        """
        Generate the INSERT SQL statement from the given parameters.
        """
        columns = ', '.join(data.keys())
        values = ', '.join(f"'{value}'" for value in data.values())
        query = f"INSERT INTO {table} ({columns}) VALUES ({values})"
        return query

    @staticmethod
    def delete(table, where=None):
        """
        Generate the DELETE SQL statement from the given parameters.
        """
        query = f"DELETE FROM {table}"
        if where:
            conditions = ' AND '.join([f"{key}='{value}'" for key, value in where.items()])
            query += f" WHERE {conditions}"
        return query

    @staticmethod
    def update(table, data, where=None):
        """
        Generate the UPDATE SQL statement from the given parameters.
        """
        set_clause = ', '.join([f"{key}='{value}'" for key, value in data.items()])
        query = f"UPDATE {table} SET {set_clause}"
        if where:
            conditions = ' AND '.join([f"{key}='{value}'" for key, value in where.items()])
            query += f" WHERE {conditions}"
        return query

# Test cases
if __name__ == "__main__":
    # Test SELECT
    select_query = SQLQueryBuilder.select('table1', columns=["col1", "col2"], where={"age": 15})
    print(select_query)  # Expected: "SELECT col1, col2 FROM table1 WHERE age='15'"

    # Test INSERT
    insert_query = SQLQueryBuilder.insert('table1', {'name': 'Test', 'age': 14})
    print(insert_query)  # Expected: "INSERT INTO table1 (name, age) VALUES ('Test', '14')"

    # Test DELETE
    delete_query = SQLQueryBuilder.delete('table1', {'name': 'Test', 'age': 14})
    print(delete_query)  # Expected: "DELETE FROM table1 WHERE name='Test' AND age='14'"

    # Test UPDATE
    update_query = SQLQueryBuilder.update('table1', {'name': 'Test2', 'age': 15}, where={'name': 'Test'})
    print(update_query)  # Expected: "UPDATE table1 SET name='Test2', age='15' WHERE name='Test'"