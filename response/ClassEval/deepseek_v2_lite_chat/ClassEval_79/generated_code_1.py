import sqlite3

class SQLGenerator:
    def __init__(self, table_name):
        self.table_name = table_name
        self.conn = sqlite3.connect(':memory:')  # Use in-memory database for simplicity
        self.cursor = self.conn.cursor()

    def select(self, fields=None, condition=None):
        select_clause = 'SELECT ' + ', '.join(fields or []) + ' FROM ' + self.table_name
        if condition:
            select_clause += ' WHERE ' + condition
        return select_clause + ';'

    def insert(self, data):
        placeholders = ', '.join('?' * len(data))
        insert_query = f'INSERT INTO {self.table_name} ( {", ".join(data.keys())}) VALUES ( {placeholders})'
        return insert_query

    def update(self, data, condition):
        update_values = ', '.join(f'{key} = ?' for key in data)
        update_query = f'UPDATE {self.table_name} SET {update_values} WHERE {condition}'
        return update_query

    def delete(self, condition):
        delete_query = f'DELETE FROM {self.table_name} WHERE {condition}'
        return delete_query

    def select_female_under_age(self, age):
        select_query = f'SELECT * FROM {self.table_name} WHERE age < ? AND gender = "female";'
        return select_query

    def select_by_age_range(self, min_age, max_age):
        select_query = f'SELECT * FROM {self.table_name} WHERE age BETWEEN ? AND ?;'
        return select_query

    def __del__(self):
        self.conn.close()

# Test cases
if __name__ == "__main__":
    sql = SQLGenerator('table1')
    print(sql.select(['field1', 'field2'], 'filed3 = value1'))
    print(sql.insert({'key1': 'value1', 'key2': 'value2'}))
    print(sql.update({'field1': 'new_value1', 'field2': 'new_value2'}, "field3 = value1"))
    print(sql.delete("field1 = value1"))
    print(sql.select_female_under_age(30))
    print(sql.select_by_age_range(20, 30))