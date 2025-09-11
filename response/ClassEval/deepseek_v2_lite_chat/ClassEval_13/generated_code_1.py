class BookManagement:
    def __init__(self):
        """
        Initialize the inventory of Book Manager.
        """
        self.inventory = {}

    def add_book(self, title, quantity=1):
        """
        Add one or several books to inventory which is sorted by book title.
        :param title: str, the book title
        :param quantity: int, default value is 1.
        """
        if title in self.inventory:
            self.inventory[title] += quantity
        else:
            self.inventory[title] = quantity

    def remove_book(self, title, quantity):
        """
        Remove one or several books from inventory which is sorted by book title.
        Raise false while get invalid input.
        :param title: str, the book title
        :param quantity: int
        """
        if title not in self.inventory:
            raise ValueError("Invalid input. The book does not exist in the inventory.")
        if self.inventory[title] < quantity:
            raise ValueError("Invalid input. Not enough books in the inventory.")
        self.inventory[title] -= quantity

    def view_inventory(self):
        """
        Get the inventory of the Book Management.
        :return self.inventory: dictionary, {title(str): quantity(int), ...}
        """
        return self.inventory

    def view_book_quantity(self, title):
        """
        Get the quantity of a book.
        :param title: str, the title of the book.
        :return quantity: the quantity of this book title. return 0 when the title does not exist in self.invenroty
        """
        return self.inventory.get(title, 0)

# Test cases
if __name__ == "__main__":
    bookManagement = BookManagement()

    # Add books
    bookManagement.add_book("book1", 5)
    bookManagement.add_book("book2", 3)
    bookManagement.add_book("book3", 2)

    # View inventory
    print(bookManagement.view_inventory())  # Should print: {'book1': 5, 'book2': 3, 'book3': 2}

    # View quantity of a book
    print(bookManagement.view_book_quantity("book1"))  # Should print: 5

    # Remove books
    bookManagement.remove_book("book1", 3)
    print(bookManagement.view_inventory())  # Should print: {'book2': 3, 'book3': 2}

    # Trying to remove a non-existent book
    try:
        bookManagement.remove_book("book4", 1)
    except ValueError as e:
        print(e)  # Should print: Invalid input. The book does not exist in the inventory.