import sqlite3

class SQLQueryBuilder:
    def __init__(self):
        self.conn = sqlite3.connect('database.db')
        self.c = self.conn.cursor()

    @staticmethod
    def select(table, columns='*', where=None):
        query = f'SELECT {", ".join(columns)} FROM {table}'
        if where is not None:
            query += ' WHERE ' + ' AND '.join(f'{key}={value}' for key, value in where.items())
        return query

    @staticmethod
    def insert(table, data):
        query = f'INSERT INTO {table} ({", ".join(data.keys())}) VALUES ({", ".join(data.values())})'
        return query

    @staticmethod
    def delete(table, where=None):
        query = f'DELETE FROM {table}'
        if where is not None:
            query += ' WHERE ' + ' AND '.join(f'{key}={value}' for key, value in where.items())
        return query

    @staticmethod
    def update(table, data, where=None):
        query = f'UPDATE {table} SET {", ".join(f"{key}={value}" for key, value in data.items())}'
        if where is not None:
            query += ' WHERE ' + ' AND '.join(f'{key}={value}' for key, value in where.items())
        return query

    def execute(self, query):
        self.c.execute(query)
        self.conn.commit()

# Test cases
if __name__ == "__main__":
    builder = SQLQueryBuilder()

    # SELECT
    query = builder.select('table1', ['col1', 'col2'], {'age': '15'})
    print(query)

    # INSERT
    data = {'name': 'Test', 'age': '14'}
    query = builder.insert('table1', data)
    print(query)

    # DELETE
    query = builder.delete('table1', {'name': 'Test', 'age': '14'})
    print(query)

    # UPDATE
    data = {'name': 'Test2', 'age': '15'}
    query = builder.update('table1', data, {'name': 'Test'})
    print(query)

    builder.conn.close()






