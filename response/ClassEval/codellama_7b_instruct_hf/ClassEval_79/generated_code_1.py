import sqlite3

class SQLGenerator:
    def __init__(self, table_name):
        self.table_name = table_name

    def select(self, fields=None, condition=None):
        if fields is None:
            fields = ["*"]
        query = "SELECT {} FROM {}".format(", ".join(fields), self.table_name)
        if condition is not None:
            query += " WHERE {}".format(condition)
        return query

    def insert(self, data):
        fields = ", ".join(data.keys())
        values = ", ".join(["'{}'".format(value) for value in data.values()])
        query = "INSERT INTO {} ({}) VALUES ({})".format(self.table_name, fields, values)
        return query

    def update(self, data, condition):
        fields = ", ".join(data.keys())
        values = ", ".join(["'{}'".format(value) for value in data.values()])
        query = "UPDATE {} SET {} WHERE {}".format(self.table_name, fields, condition)
        return query

    def delete(self, condition):
        query = "DELETE FROM {} WHERE {}".format(self.table_name, condition)
        return query

    def select_female_under_age(self, age):
        query = self.select(["*"], "gender = 'female' AND age < {}".format(age))
        return query

    def select_by_age_range(self, min_age, max_age):
        query = self.select(["*"], "age BETWEEN {} AND {}".format(min_age, max_age))
        return query
    
if __name__ == "__main__":
    # 初始化SQL生成器，指定表名为"users"
    sql_gen = SQLGenerator("users")

    # 测试SELECT语句（查询所有字段）
    print("1. 查询所有字段:")
    print(sql_gen.select())
    print()

    # 测试SELECT语句（指定字段和条件）
    print("2. 查询指定字段和条件:")
    print(sql_gen.select(fields=["name", "age"], condition="gender = 'male'"))
    print()

    # 测试INSERT语句
    print("3. 插入数据:")
    insert_data = {"name": "Alice", "age": 25, "gender": "female"}
    print(sql_gen.insert(insert_data))
    print()

    # 测试UPDATE语句
    print("4. 更新数据:")
    update_data = {"age": 26, "gender": "female"}
    print(sql_gen.update(update_data, "name = 'Alice'"))
    print()

    # 测试DELETE语句
    print("5. 删除数据:")
    print(sql_gen.delete("age > 30"))
    print()

    # 测试查询特定条件的女性（年龄小于30）
    print("6. 查询年龄小于30的女性:")
    print(sql_gen.select_female_under_age(30))
    print()

    # 测试查询年龄范围内的记录（20到30岁）
    print("7. 查询20到30岁的记录:")
    print(sql_gen.select_by_age_range(20, 30))
