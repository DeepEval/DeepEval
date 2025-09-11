import sqlite3

class BookManagementDB:
    """
    This is a database class as a book management system, used to handle the operations of adding, removing, updating, and searching books.
    """

    def __init__(self, db_name):
        """
        Initializes the class by creating a database connection and cursor, 
        and creates the book table if it does not already exist
        :param db_name: str, the name of db file
        """
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """
        Creates the book table in the database if it does not already exist.
        >>> book_db = BookManagementDB("test.db")
        >>> book_db.create_table()
        """
        # Create table query
        query = """
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT NOT NULL,
                availability INTEGER NOT NULL DEFAULT 1
            )
        """
        self.cursor.execute(query)
        self.connection.commit()

    def add_book(self, title, author):
        """
        Adds a book to the database with the specified title and author, 
        setting its availability to 1 as free to borrow.
        :param title: str, book title
        :param author: str, author name
        >>> book_db = BookManagementDB("test.db")
        >>> book_db.create_table()
        >>> book_db.add_book('book1', 'author')
        """
        # Insert query
        query = """
            INSERT INTO books (title, author)
            VALUES (?,?)
        """
        self.cursor.execute(query, (title, author))
        self.connection.commit()
        return self.cursor.lastrowid

    def remove_book(self, book_id):
        """
        Removes a book from the database based on the given book ID.
        :param book_id: int
        >>> book_db = BookManagementDB("test.db")
        >>> book_db.remove_book(1)
        """
        # Delete query
        query = """
            DELETE FROM books
            WHERE id =?
        """
        self.cursor.execute(query, (book_id,))
        self.connection.commit()

    def borrow_book(self, book_id):
        """
        Marks a book as borrowed in the database based on the given book ID.
        :param book_id: int
        >>> book_db = BookManagementDB("test.db")
        >>> book_db.borrow_book(1)
        """
        # Update query
        query = """
            UPDATE books
            SET availability = 0
            WHERE id =?
        """
        self.cursor.execute(query, (book_id,))
        self.connection.commit()

    def return_book(self, book_id):
        """
        Marks a book as returned in the database based on the given book ID.
        :param book_id: int
        >>> book_db = BookManagementDB("test.db")
        >>> book_db.return_book(1)
        """
        # Update query
        query = """
            UPDATE books
            SET availability = 1
            WHERE id =?
        """
        self.cursor.execute(query, (book_id,))
        self.connection.commit()

    def search_books(self):
        """
        Retrieves all books from the database and returns their information.
        :return books: list[tuple], the information of all books in database
        >>> book_db = BookManagementDB("test.db")
        >>> book_db.search_books()
        [(1, 'book1', 'author', 1)]
        """
        # Select query
        query = """
            SELECT id, title, author, availability
            FROM books
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

if __name__ == "__main__":
    book_db = BookManagementDB("test.db")
    book_db.create_table()

    # Test case for add_book
    book_id = book_db.add_book('book1', 'author')
    print(f"Added book with id: {book_id}")

    # Test case for remove_book
    book_db.remove_book(book_id)
    print(f"Removed book with id: {book_id}")

    # Test case for borrow_book
    book_db.add_book('book2', 'author')
    book_id = 2
    book_db.borrow_book(book_id)
    print(f"Borrowed book with id: {book_id}")

    # Test case for return_book
    book_db.return_book(book_id)
    print(f"Returned book with id: {book_id}")

    # Test case for search_books
    books = book_db.search_books()
    for book in books:
        print(book)

    # Close the database connection
    book_db.connection.close()