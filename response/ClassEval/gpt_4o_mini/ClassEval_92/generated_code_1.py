import sqlite3

class UserLoginDB:
    """
    This is a database management class for user login verification, providing functions for inserting user information,
    searching user information, deleting user information, and validating user login.
    """

    def __init__(self, db_name):
        """
        Initializes the UserLoginDB object with the specified database name.
        :param db_name: str, the name of the SQLite database.
        """
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """
        Creates the "users" table if it does not exist.
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
        try:
            self.cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            self.connection.commit()
        except sqlite3.IntegrityError:
            print(f"User {username} already exists.")

    def search_user_by_username(self, username):
        """
        Searches for users in the "users" table by username.
        :param username: str, the username of the user to search for.
        :return: list of tuples, the rows from the "users" table that match the search criteria.
        """
        self.cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        return self.cursor.fetchall()

    def delete_user_by_username(self, username):
        """
        Deletes a user from the "users" table by username.
        :param username: str, the username of the user to delete.
        :return: None
        """
        self.cursor.execute('DELETE FROM users WHERE username = ?', (username,))
        self.connection.commit()

    def validate_user_login(self, username, password):
        """
        Determine whether the user can log in, that is, the user is in the database and the password is correct
        :param username: str, the username of the user to validate.
        :param password: str, the password of the user to validate.
        :return: bool, representing whether the user can log in correctly
        """
        self.cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
        result = self.cursor.fetchone()
        return result is not None and result[0] == password

    def close(self):
        """Closes the database connection."""
        self.connection.close()


if __name__ == "__main__":
    # Testing UserLoginDB
    user_db = UserLoginDB("user_database.db")

    # Test insert_user
    user_db.insert_user('user1', 'pass1')  # Insert user
    user_db.insert_user('user1', 'pass1')  # Try to insert the same user again

    # Test search_user_by_username
    result = user_db.search_user_by_username('user1')
    print(f"Search user1: {result}")  # Should return user1

    # Test validate_user_login
    login_success = user_db.validate_user_login('user1', 'pass1')
    print(f"Login user1 success: {login_success}")  # Should return True

    # Test delete_user_by_username
    user_db.delete_user_by_username('user1')
    result_after_delete = user_db.search_user_by_username('user1')
    print(f"Search user1 after deletion: {result_after_delete}")  # Should return []

    # Clean up
    user_db.close()