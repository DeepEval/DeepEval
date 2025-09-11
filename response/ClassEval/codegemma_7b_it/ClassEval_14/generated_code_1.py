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
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            availability INTEGER NOT NULL
        )""")
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
        self.cursor.execute("INSERT INTO books (title, author, availability) VALUES (?, ?, ?)", (title, author, 1))
        self.connection.commit()

    def remove_book(self, book_id):
        """
        Removes a book from the database based on the given book ID.
        :param book_id: int
        >>> book_db = BookManagementDB("test.db")
        >>> book_db.remove_book(1)
        """
        self.cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
        self.connection.commit()

    def borrow_book(self, book_id):
        """
        Marks a book as borrowed in the database based on the given book ID.
        :param book_id: int
        >>> book_db = BookManagementDB("test.db")
        >>> book_db.borrow_book(1)
        """
        self.cursor.execute("UPDATE books SET availability = 0 WHERE id = ?", (book_id,))
        self.connection.commit()

    def return_book(self, book_id):
        """
        Marks a book as returned in the database based on the given book ID.
        :param book_id: int
        >>> book_db = BookManagementDB("test.db")
        >>> book_db.return_book(1)
        """
        self.cursor.execute("UPDATE books SET availability = 1 WHERE id = ?", (book_id,))
        self.connection.commit()

    def search_books(self):
        """
        Retrieves all books from the database and returns their information.
        :return books: list[tuple], the information of all books in database
        >>> book_db.search_books()
        [(1, 'book1', 'author', 1)]
        """
        self.cursor.execute("SELECT * FROM books")
        books = self.cursor.fetchall()
        return books
    
if __name__ == "__main__":
    book_db = BookManagementDB("test.db")
    book_db.create_table()
    book_db.add_book('book1', 'author')
    book_db.add_book('book2', 'author')
    book_db.add_book('book3', 'author')
    book_db.add_book('book4', 'author')
    book_db.add_book('book5', 'author')
    book_db.add_book('book6', 'author')
    book_db.add_book('book7', 'author')
    book_db.add_book('book8', 'author')
    book_db.add_book('book9', 'author')
    book_db.add_book('book10', 'author')
    book_db.add_book('book11', 'author')
    book_db.add_book('book12', 'author')
    book_db.add_book('book13', 'author')
    book_db.add_book('book14', 'author')
    book_db.add_book('book15', 'author')
    book_db.add_book('book16', 'author')
    book_db.add_book('book17', 'author')
    book_db.add_book('book18', 'author')
    book_db.add_book('book19', 'author')
    book_db.add_book('book20', 'author')
    book_db.add_book('book21', 'author')
    book_db.add_book('book22', 'author')
    book_db.add_book('book23', 'author')
    book_db.add_book('book24', 'author')
    book_db.add_book('book25', 'author')
    book_db.add_book('book26', 'author')
    book_db.add_book('book27', 'author')
    book_db.add_book('book28', 'author')
    book_db.add_book('book29', 'author')
    book_db.add_book('book30', 'author')
    book_db.add_book('book31', 'author')
    book_db.add_book('book32', 'author')
    book_db.add_book('book33', 'author')
    book_db.add_book('book34', 'author')
    book_db.add_book('book35', 'author')
    book_db.add_book('book36', 'author')
    book_db.add_book('book37', 'author')
    book_db.add_book('book38', 'author')
    book_db.add_book('book39', 'author')
    book_db.add_book('book40', 'author')
    book_db.add_book('book41', 'author')
    book_db.add_book('book42', 'author')
    book_db.add_book('book43', 'author')
    book_db.add_book('book44', 'author')
    book_db.add_book('book45', 'author')
    book_db.add_book('book46', 'author')
    book_db.add_book('book47', 'author')
    book_db.add_book('book48', 'author')
    book_db.add_book('book49', 'author')
    book_db.add_book('book50', 'author')
    
    print(book_db.search_books())
    
    book_db.remove_book(1)
    
    print(book_db.search_books())
    
    book_db.borrow_book(2)
    
    print(book_db.search_books())
    
    book_db.return_book(2)
    
    print(book_db.search_books())
    
    book_db.connection.close()