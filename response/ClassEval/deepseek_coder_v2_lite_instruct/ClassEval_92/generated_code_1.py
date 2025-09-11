import sqlite3

class UserLoginDB:
    """
    This is a database management class for user login verification, providing functions for inserting user information, searching user information, deleting user information, and validating user login.
    """

    def __init__(self, db_name):
        """
        Initializes the UserLoginDB object with the specified database name.
        :param db_name: str, the name of the SQLite database.
        """
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def create_table(self):
        """
        Creates the "users" table if it doesn't exist.
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        ''')
        self.connection.commit()

    def insert_user(self, username, password):
        """
        Inserts a new user into the "users" table.
        :param username: str, the username of the user.
        :param password: str, the password of the user.
        :return: None
        """
        self.cursor.execute('''
            INSERT INTO users (username, password) VALUES (?, ?)
            ON CONFLICT(username) DO UPDATE SET password=excluded.password
        ''', (username, password))
        self.connection.commit()

    def search_user_by_username(self, username):
        """
        Searches for users in the "users" table by username.
        :param username: str, the username of the user to search for.
        :return: list of tuples, the rows from the "users" table that match the search criteria.
        """
        self.cursor.execute('''
            SELECT * FROM users WHERE username = ?
        ''', (username,))
        return self.cursor.fetchall()

    def delete_user_by_username(self, username):
        """
        Deletes a user from the "users" table by username.
        :param username: str, the username of the user to delete.
        :return: None
        """
        self.cursor.execute('''
            DELETE FROM users WHERE username = ?
        ''', (username,))
        self.connection.commit()

    def validate_user_login(self, username, password):
        """
        Determine whether the user can log in, that is, the user is in the database and the password is correct
        :param username: str, the username of the user to validate.
        :param password: str, the password of the user to validate.
        :return: bool, representing whether the user can log in correctly
        """
        self.cursor.execute('''
            SELECT * FROM users WHERE username = ? AND password = ?
        ''', (username, password))
        return self.cursor.fetchone() is not None

if __name__ == "__main__":
    user_db = UserLoginDB("user_database.db")
    user_db.create_table()

    # Test cases
    print("Insert User Test:")
    user_db.insert_user('user1', 'pass1')
    print(user_db.search_user_by_username('user1'))  # Should return [('user1', 'pass1')]

    print("\nSearch User by Username Test:")
    result = user_db.search_user_by_username('user1')
    print(len(result))  # Should return 1

    print("\nDelete User by Username Test:")
    user_db.delete_user_by_username('user1')
    result = user_db.search_user_by_username('user1')
    print(len(result))  # Should return 0

    print("\nValidate User Login Test:")
    user_db.insert_user('user1', 'pass1')
    print(user_db.validate_user_login('user1', 'pass1'))  # Should return True
    print(user_db.validate_user_login('user1', 'wrongpass'))  # Should return False