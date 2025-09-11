import sqlite3

class UserLoginDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def create_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
        self.connection.commit()

    def insert_user(self, username, password):
        self.cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        self.connection.commit()

    def search_user_by_username(self, username):
        self.cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        result = self.cursor.fetchall()
        return result

    def delete_user_by_username(self, username):
        self.cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        self.connection.commit()

    def validate_user_login(self, username, password):
        self.cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = self.cursor.fetchone()
        if result is None:
            return False
        return result[0] == password

if __name__ == "__main__":
    db = UserLoginDB("user_database.db")
    db.create_table()
    db.insert_user("user1", "pass1")
    result = db.search_user_by_username("user1")
    print(result)  # [('user1', 'pass1')]
    db.delete_user_by_username("user1")
    print(db.search_user_by_username("user1"))  # []
    print(db.validate_user_login("user1", "pass1"))  # False
    print(db.validate_user_login("user1", "pass2"))  # True