import unittest

class SQLQueryBuilder:
    """
    This class provides to build SQL queries, including SELECT, INSERT, UPDATE,
    and DELETE statements. 
    """

    @staticmethod
    def select(table, columns='*', where=None):
        """
        Generate the SELECT SQL statement from the given parameters.
        :param table: str, the query table in database.
        :param columns: list of str, ['col1', 'col2'].
        :param where: dict, {key1: value1, key2: value2...}. The query condition.
        return query: str, the SQL query statement.
        >>> SQLQueryBuilder.select('table1', columns = ["col1","col2"], where = {"age": 15})
        "SELECT col1, col2 FROM table1 WHERE age='15'"
        """
        query = "SELECT " + ", ".join(columns) + " FROM " + table
        if where:
            query += " WHERE "
            query += " AND ".join([f"{key}='{value}'" for key, value in where.items()])
        return query

    @staticmethod
    def insert(table, data):
        """
        Generate the INSERT SQL statement from the given parameters.
        :param table: str, the table to be inserted in database.
        :param data: dict, the key and value in SQL insert statement
        :return query: str, the SQL insert statement.
        >>> SQLQueryBuilder.insert('table1', {'name': 'Test', 'age': 14})
        "INSERT INTO table1 (name, age) VALUES ('Test', '14')"
        """
        columns = ", ".join(data.keys())
        values = ", ".join([f"'{value}'" for value in data.values()])
        query = f"INSERT INTO {table} ({columns}) VALUES ({values})"
        return query

    @staticmethod
    def delete(table, where=None):
        """
        Generate the DELETE SQL statement from the given parameters.
        :param table: str, the table that will be excuted with DELETE operation in database
        :param where: dict, {key1: value1, key2: value2...}. The query condition.
        :return query: str, the SQL delete statement.
        >>> SQLQueryBuilder.delete('table1', {'name': 'Test', 'age': 14})
        "DELETE FROM table1 WHERE name='Test' AND age='14'"
        """
        query = f"DELETE FROM {table}"
        if where:
            query += " WHERE "
            query += " AND ".join([f"{key}='{value}'" for key, value in where.items()])
        return query

    @staticmethod
    def update(table, data, where=None):
        """
        Generate the UPDATE SQL statement from the given parameters.
        :param table: str, the table that will be excuted with UPDATE operation in database
        :param data: dict, the key and value in SQL update statement
        :param where: dict, {key1: value1, key2: value2...}. The query condition.
        >>> SQLQueryBuilder.update('table1', {'name': 'Test2', 'age': 15}, where = {'name':'Test'})
        "UPDATE table1 SET name='Test2', age='15' WHERE name='Test'"
        """
        query = f"UPDATE {table} SET "
        query += ", ".join([f"{key}='{value}'" for key, value in data.items()])
        if where:
            query += " WHERE "
            query += " AND ".join([f"{key}='{value}'" for key, value in where.items()])
        return query


class TestSQLQueryBuilder(unittest.TestCase):

    def test_select(self):
        self.assertEqual(SQLQueryBuilder.select('table1', columns=["col1", "col2"], where={"age": 15}),
                         "SELECT col1, col2 FROM table1 WHERE age='15'")

    def test_insert(self):
        self.assertEqual(SQLQueryBuilder.insert('table1', {'name': 'Test', 'age': 14}),
                         "INSERT INTO table1 (name, age) VALUES ('Test', '14')")

    def test_delete(self):
        self.assertEqual(SQLQueryBuilder.delete('table1', {'name': 'Test', 'age': 14}),
                         "DELETE FROM table1 WHERE name='Test' AND age='14'")

    def test_update(self):
        self.assertEqual(SQLQueryBuilder.update('table1', {'name': 'Test2', 'age': 15}, where={'name': 'Test'}),
                         "UPDATE table1 SET name='Test2', age='15' WHERE name='Test'")


if __name__ == "__main__":
    unittest.main()