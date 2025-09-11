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
        :return query: str, the SQL query statement.
        """
        if isinstance(columns, str):
            columns = [columns]
        query = f"SELECT {', '.join(columns)} FROM {table}"
        if where:
            conditions = [f"{key}='{value}'" for key, value in where.items()]
            query += f" WHERE {' AND '.join(conditions)}"
        return query

    @staticmethod
    def insert(table, data):
        """
        Generate the INSERT SQL statement from the given parameters.
        :param table: str, the table to be inserted in database.
        :param data: dict, the key and value in SQL insert statement
        :return query: str, the SQL insert statement.
        """
        columns = ', '.join(data.keys())
        values = ', '.join(f"'{value}'" for value in data.values())
        query = f"INSERT INTO {table} ({columns}) VALUES ({values})"
        return query

    @staticmethod
    def delete(table, where=None):
        """
        Generate the DELETE SQL statement from the given parameters.
        :param table: str, the table that will be executed with DELETE operation in database
        :param where: dict, {key1: value1, key2: value2 ...}. The query condition.
        :return query: str, the SQL delete statement.
        """
        query = f"DELETE FROM {table}"
        if where:
            conditions = [f"{key}='{value}'" for key, value in where.items()]
            query += f" WHERE {' AND '.join(conditions)}"
        return query

    @staticmethod
    def update(table, data, where=None):
        """
        Generate the UPDATE SQL statement from the given parameters.
        :param table: str, the table that will be executed with UPDATE operation in database
        :param data: dict, the key and value in SQL update statement
        :param where: dict, {key1: value1, key2: value2 ...}. The query condition.
        :return query: str, the SQL update statement.
        """
        set_clauses = [f"{key}='{value}'" for key, value in data.items()]
        query = f"UPDATE {table} SET {', '.join(set_clauses)}"
        if where:
            conditions = [f"{key}='{value}'" for key, value in where.items()]
            query += f" WHERE {' AND '.join(conditions)}"
        return query

# Test cases
if __name__ == "__main__":
    # Test SELECT
    print(SQLQueryBuilder.select('table1', columns=["col1", "col2"], where={"age": 15}))
    # Expected output: "SELECT col1, col2 FROM table1 WHERE age='15'"

    # Test INSERT
    print(SQLQueryBuilder.insert('table1', {'name': 'Test', 'age': 14}))
    # Expected output: "INSERT INTO table1 (name, age) VALUES ('Test', '14')"

    # Test DELETE
    print(SQLQueryBuilder.delete('table1', {'name': 'Test', 'age': 14}))
    # Expected output: "DELETE FROM table1 WHERE name='Test' AND age='14'"

    # Test UPDATE
    print(SQLQueryBuilder.update('table1', {'name': 'Test2', 'age': 15}, where={'name': 'Test'}))
    # Expected output: "UPDATE table1 SET name='Test2', age='15' WHERE name='Test'"