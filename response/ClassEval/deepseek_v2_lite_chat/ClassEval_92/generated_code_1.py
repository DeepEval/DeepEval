import sqlite3

class UserLoginDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """Creates a table for user login information."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT
            )
        ''').close()

    def insert_user(self, username, password):
        """Inserts a new user into the "users" table."""
        self.cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password)).close()
        self.connection.commit()

    def search_user_by_username(self, username):
        """Searches for users in the "users" table by username."""
        self.cursor.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchall()
        return self.cursor.fetchall()

    def delete_user_by_username(self, username):
        """Deletes a user from the "users" table by username."""
        self.cursor.execute('DELETE FROM users WHERE username = ?', (username,)).close()
        self.connection.commit()

    def validate_user_login(self, username, password):
        """Determine whether the user can log in, that is, the user is in the database and the password is correct."""
        self.cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
        return bool(self.cursor.fetchone())

# Example usage
if __name__ == "__main__":
    user_db = UserLoginDB("user_database.db")
    user_db.create_table()
    user_db.insert_user('user1', 'pass1')
    result = user_db.search_user_by_username('user1')
    print(len(result) == 1)  # Should print True
    deleted = user_db.delete_user_by_username('user1')
    print(deleted)  # Should print None
    login_status = user_db.validate_user_login('user1', 'pass1')
    print(login_status)  # Should print True