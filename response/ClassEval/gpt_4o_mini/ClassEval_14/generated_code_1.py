import sqlite3

class BookManagementDB:
    """
    This is a database class as a book management system, used to handle the operations of adding, removing, updating, and searching books.
    """

    def __init__(self, db_name):
        """
        Initializes the class by creating a database connection and cursor, 
        and creates the book table if it does not already exist.
        :param db_name: str, the name of db file
        """
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """
        Creates the book table in the database if it does not already exist.
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT NOT NULL,
                available INTEGER NOT NULL DEFAULT 1
            )
        ''')
        self.connection.commit()

    def add_book(self, title, author):
        """
        Adds a book to the database with the specified title and author, 
        setting its availability to 1 as free to borrow.
        :param title: str, book title
        :param author: str, author name
        """
        self.cursor.execute('''
            INSERT INTO books (title, author) VALUES (?, ?)
        ''', (title, author))
        self.connection.commit()

    def remove_book(self, book_id):
        """
        Removes a book from the database based on the given book ID.
        :param book_id: int
        """
        self.cursor.execute('''
            DELETE FROM books WHERE id = ?
        ''', (book_id,))
        self.connection.commit()

    def borrow_book(self, book_id):
        """
        Marks a book as borrowed in the database based on the given book ID.
        :param book_id: int
        """
        self.cursor.execute('''
            UPDATE books SET available = 0 WHERE id = ?
        ''', (book_id,))
        self.connection.commit()

    def return_book(self, book_id):
        """
        Marks a book as returned in the database based on the given book ID.
        :param book_id: int
        """
        self.cursor.execute('''
            UPDATE books SET available = 1 WHERE id = ?
        ''', (book_id,))
        self.connection.commit()

    def search_books(self):
        """
        Retrieves all books from the database and returns their information.
        :return books: list[tuple], the information of all books in database
        """
        self.cursor.execute('SELECT * FROM books')
        return self.cursor.fetchall()

    def close(self):
        """
        Closes the database connection.
        """
        self.connection.close()


if __name__ == "__main__":
    # Test case for the BookManagementDB class
    book_db = BookManagementDB("test.db")
    
    # Test add_book
    book_db.add_book('Book 1', 'Author 1')
    book_db.add_book('Book 2', 'Author 2')
    print("Books after adding:", book_db.search_books())
    
    # Test borrow_book
    book_db.borrow_book(1)
    print("Books after borrowing Book 1:", book_db.search_books())
    
    # Test return_book
    book_db.return_book(1)
    print("Books after returning Book 1:", book_db.search_books())
    
    # Test remove_book
    book_db.remove_book(2)
    print("Books after removing Book 2:", book_db.search_books())

    # Close the database connection
    book_db.close()