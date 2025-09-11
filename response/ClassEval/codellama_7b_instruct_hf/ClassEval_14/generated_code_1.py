import sqlite3

class BookManagementDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS books (id INTEGER PRIMARY KEY, title TEXT, author TEXT, availability INTEGER)")
        self.connection.commit()

    def add_book(self, title, author):
        self.cursor.execute("INSERT INTO books (title, author, availability) VALUES (?, ?, ?)", (title, author, 1))
        self.connection.commit()

    def remove_book(self, book_id):
        self.cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
        self.connection.commit()

    def borrow_book(self, book_id):
        self.cursor.execute("UPDATE books SET availability = 0 WHERE id = ?", (book_id,))
        self.connection.commit()

    def return_book(self, book_id):
        self.cursor.execute("UPDATE books SET availability = 1 WHERE id = ?", (book_id,))
        self.connection.commit()

    def search_books(self):
        self.cursor.execute("SELECT * FROM books")
        books = self.cursor.fetchall()
        return books

if __name__ == "__main__":
    db = BookManagementDB("test.db")
    db.create_table()
    db.add_book("book1", "author")
    db.borrow_book(1)
    db.return_book(1)
    books = db.search_books()
    print(books)