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
        if title in self.inventory:
            self.inventory[title] += quantity
        else:
            self.inventory[title] = quantity
        # Ensure the inventory is sorted by book title
        self.inventory = dict(sorted(self.inventory.items()))

    def remove_book(self, title, quantity):
        """
        Remove one or several books from inventory which is sorted by book title.
        Raise false while get invalid input.
        :param title: str, the book title
        :param quantity: int
        """
        if title not in self.inventory or quantity > self.inventory[title]:
            return False
        self.inventory[title] -= quantity
        if self.inventory[title] == 0:
            del self.inventory[title]
        # Ensure the inventory is sorted by book title
        self.inventory = dict(sorted(self.inventory.items()))
        return True

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
        :return quantity: the quantity of this book title. return 0 when the title does not exist in self.inventory
        """
        return self.inventory.get(title, 0)

# Test cases
if __name__ == "__main__":
    bookManagement = BookManagement()
    print("Initial inventory:", bookManagement.view_inventory())
    
    # Test add_book
    bookManagement.add_book("book1", 1)
    bookManagement.add_book("book2", 1)
    print("After adding books:", bookManagement.view_inventory())
    
    # Test remove_book
    bookManagement.remove_book("book1", 1)
    print("After removing one book1:", bookManagement.view_inventory())
    
    # Test view_book_quantity
    print("Quantity of book1:", bookManagement.view_book_quantity("book1"))
    print("Quantity of book3:", bookManagement.view_book_quantity("book3"))