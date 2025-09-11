class BookManagement:
    """
    This is a class as managing books system, which supports to add and remove books from the inventory dict,
    view the inventory, and check the quantity of a specific book.
    """

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
        if quantity < 1:
            raise ValueError("Quantity must be at least 1.")
        
        if title in self.inventory:
            self.inventory[title] += quantity
        else:
            self.inventory[title] = quantity

        self.inventory = dict(sorted(self.inventory.items()))  # Sort the inventory by title

    def remove_book(self, title, quantity):
        """
        Remove one or several books from inventory which is sorted by book title.
        Raise an error when getting invalid input.
        :param title: str, the book title
        :param quantity: int
        """
        if quantity < 1:
            raise ValueError("Quantity must be at least 1.")
        
        if title not in self.inventory:
            raise ValueError("Book not found in inventory.")
        
        if self.inventory[title] < quantity:
            raise ValueError("Not enough quantity to remove.")

        self.inventory[title] -= quantity
        if self.inventory[title] == 0:
            del self.inventory[title]

    def view_inventory(self):
        """
        Get the inventory of the Book Management.
        :return: dictionary, {title(str): quantity(int), ...}
        """
        return self.inventory

    def view_book_quantity(self, title):
        """
        Get the quantity of a book.
        :param title: str, the title of the book.
        :return quantity: the quantity of this book title. return 0 when the title does not exist in self.inventory
        """
        return self.inventory.get(title, 0)


# Test cases
if __name__ == "__main__":
    bookManagement = BookManagement()

    # Test adding books
    bookManagement.add_book("book1", 2)
    print(bookManagement.view_inventory())  # {'book1': 2}
    bookManagement.add_book("book2", 3)
    print(bookManagement.view_inventory())  # {'book1': 2, 'book2': 3}
    bookManagement.add_book("book1", 1)
    print(bookManagement.view_inventory())  # {'book1': 3, 'book2': 3}

    # Test removing books
    bookManagement.remove_book("book1", 1)
    print(bookManagement.view_inventory())  # {'book1': 2, 'book2': 3}
    bookManagement.remove_book("book2", 3)
    print(bookManagement.view_inventory())  # {'book1': 2}

    # Test viewing book quantity
    print(bookManagement.view_book_quantity("book1"))  # 2
    print(bookManagement.view_book_quantity("book2"))  # 0

    # Test removing more than available
    try:
        bookManagement.remove_book("book1", 3)  # Should raise an error
    except ValueError as e:
        print(e)  # Not enough quantity to remove.

    # Test adding invalid quantity
    try:
        bookManagement.add_book("book3", -1)  # Should raise an error
    except ValueError as e:
        print(e)  # Quantity must be at least 1.