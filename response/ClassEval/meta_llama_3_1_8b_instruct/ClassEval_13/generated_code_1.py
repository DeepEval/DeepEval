class BookManagement:
    """
    This is a class as managing books system, which supports to add and remove books from the inventory dict, view the inventory, and check the quantity of a specific book.
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
        if not isinstance(title, str) or not isinstance(quantity, int):
            raise ValueError("Invalid input: title must be a string and quantity must be an integer.")
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
        if not isinstance(title, str) or not isinstance(quantity, int):
            raise ValueError("Invalid input: title must be a string and quantity must be an integer.")
        if title not in self.inventory:
            raise ValueError("Book not found in inventory.")
        if quantity > self.inventory[title]:
            raise ValueError("Quantity to remove is more than the available quantity.")
        self.inventory[title] -= quantity
        if self.inventory[title] == 0:
            del self.inventory[title]

    def view_inventory(self):
        """
        Get the inventory of the Book Management.
        :return self.inventory: dictionary, {title(str): quantity(int),...}
        """
        return self.inventory

    def view_book_quantity(self, title):
        """
        Get the quantity of a book.
        :param title: str, the title of the book.
        :return quantity: the quantity of this book title. return 0 when the title does not exist in self.invenroty
        """
        return self.inventory.get(title, 0)

if __name__ == "__main__":
    book_management = BookManagement()
    
    # Test case for add_book method
    print("Test case for add_book method:")
    book_management.add_book("book1", 1)
    book_management.add_book("book2", 2)
    print(book_management.view_inventory())  # Expected output: {'book1': 1, 'book2': 2}
    book_management.add_book("book1", 3)
    print(book_management.view_inventory())  # Expected output: {'book1': 4, 'book2': 2}
    
    # Test case for remove_book method
    print("\nTest case for remove_book method:")
    book_management.remove_book("book2", 1)
    print(book_management.view_inventory())  # Expected output: {'book1': 4}
    book_management.remove_book("book1", 4)
    print(book_management.view_inventory())  # Expected output: {}
    
    # Test case for view_inventory method
    print("\nTest case for view_inventory method:")
    print(book_management.view_inventory())  # Expected output: {}
    
    # Test case for view_book_quantity method
    print("\nTest case for view_book_quantity method:")
    print(book_management.view_book_quantity("book1"))  # Expected output: 0
    book_management.add_book("book1", 1)
    print(book_management.view_book_quantity("book1"))  # Expected output: 1
    print(book_management.view_book_quantity("book2"))  # Expected output: 0