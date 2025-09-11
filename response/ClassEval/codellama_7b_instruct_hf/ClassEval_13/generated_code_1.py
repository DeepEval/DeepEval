import collections

class BookManagement:
    def __init__(self):
        self.inventory = collections.defaultdict(int)

    def add_book(self, title, quantity=1):
        self.inventory[title] += quantity

    def remove_book(self, title, quantity):
        self.inventory[title] -= quantity

    def view_inventory(self):
        return dict(self.inventory)

    def view_book_quantity(self, title):
        return self.inventory.get(title, 0)

if __name__ == "__main__":
    book_management = BookManagement()
    book_management.add_book("book1", 1)
    book_management.add_book("book2", 1)
    print(book_management.view_inventory())  # {'book1': 1, 'book2': 1}
    print(book_management.view_book_quantity("book3"))  # 0