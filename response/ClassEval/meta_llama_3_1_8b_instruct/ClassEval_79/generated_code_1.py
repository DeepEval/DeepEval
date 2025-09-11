import unittest

class SQLGenerator:
    """
    This class generates SQL statements for common operations on a table, such as SELECT, INSERT, UPDATE, and DELETE.
    """

    def __init__(self, table_name):
        """
        Initialize the table name.
        :param table_name: str
        """
        self.table_name = table_name

    def select(self, fields=None, condition=None):
        """
        Generates a SELECT SQL statement based on the specified fields and conditions.
        :param fields: list, optional. Default is None. The list of fields to be queried.
        :param condition: str, optional. Default is None. The condition expression for the query.
        :return: str. The generated SQL statement.
        """
        if fields is None:
            fields = "*"
        fields_str = ", ".join(fields)
        if condition is None:
            return f"SELECT {fields_str} FROM {self.table_name};"
        else:
            return f"SELECT {fields_str} FROM {self.table_name} WHERE {condition};"

    def insert(self, data):
        """
        Generates an INSERT SQL statement based on the given data.
        :param data: dict. The data to be inserted, in dictionary form where keys are field names and values are field values.
        :return: str. The generated SQL statement.
        """
        fields = list(data.keys())
        values = list(data.values())
        fields_str = ", ".join(fields)
        values_str = ", ".join(["'" + value + "'" for value in values])
        return f"INSERT INTO {self.table_name} ({fields_str}) VALUES ({values_str});"

    def update(self, data, condition):
        """
        Generates an UPDATE SQL statement based on the given data and condition.
        :param data: dict. The data to be updated, in dictionary form where keys are field names and values are new field values.
        :param condition: str. The condition expression for the update.
        :return: str. The generated SQL statement.
        """
        updates = []
        for field, value in data.items():
            updates.append(f"{field} = '{value}'")
        updates_str = ", ".join(updates)
        return f"UPDATE {self.table_name} SET {updates_str} WHERE {condition};"

    def delete(self, condition):
        """
        Generates a DELETE SQL statement based on the given condition.
        :param condition: str. The condition expression for the delete.
        :return: str. The generated SQL statement.
        """
        return f"DELETE FROM {self.table_name} WHERE {condition};"

    def select_female_under_age(self, age):
        """
        Generates a SQL statement to select females under a specified age.
        :param age: int. The specified age.
        :return: str. The generated SQL statement.
        """
        return f"SELECT * FROM {self.table_name} WHERE age < {age} AND gender = 'female';"

    def select_by_age_range(self, min_age, max_age):
        """
        Generates a SQL statement to select records within a specified age range.
        :param min_age: int. The minimum age.
        :param max_age: int. The maximum age.
        :return: str. The generated SQL statement.
        """
        return f"SELECT * FROM {self.table_name} WHERE age BETWEEN {min_age} AND {max_age};"

class TestSQLGenerator(unittest.TestCase):

    def test_select(self):
        sql = SQLGenerator('table1')
        self.assertEqual(sql.select(), "SELECT * FROM table1;")
        self.assertEqual(sql.select(['field1', 'field2']), "SELECT field1, field2 FROM table1;")
        self.assertEqual(sql.select(fields=['field1', 'field2'], condition="field3 = 'value1'"), "SELECT field1, field2 FROM table1 WHERE field3 = 'value1';")

    def test_insert(self):
        sql = SQLGenerator('table1')
        self.assertEqual(sql.insert({'field1': 'value1', 'field2': 'value2'}), "INSERT INTO table1 (field1, field2) VALUES ('value1', 'value2');")

    def test_update(self):
        sql = SQLGenerator('table1')
        self.assertEqual(sql.update({'field1': 'new_value1', 'field2': 'new_value2'}, "field3 = 'value1'"), "UPDATE table1 SET field1 = 'new_value1', field2 = 'new_value2' WHERE field3 = 'value1';")

    def test_delete(self):
        sql = SQLGenerator('table1')
        self.assertEqual(sql.delete("field1 = 'value1'"), "DELETE FROM table1 WHERE field1 = 'value1';")

    def test_select_female_under_age(self):
        sql = SQLGenerator('table1')
        self.assertEqual(sql.select_female_under_age(30), "SELECT * FROM table1 WHERE age < 30 AND gender = 'female';")

    def test_select_by_age_range(self):
        sql = SQLGenerator('table1')
        self.assertEqual(sql.select_by_age_range(20, 30), "SELECT * FROM table1 WHERE age BETWEEN 20 AND 30;")

if __name__ == "__main__":
    unittest.main()