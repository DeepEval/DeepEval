class ShoppingCart:
    """
    The class manages items, their prices, quantities, and allows to for add, remove, view items, and calculate the total price.
    """

    def __init__(self):
        """
        Initialize the items representing the shopping list as an empty dictionary
        """
        self.items = {}

    def add_item(self, item, price, quantity=1):
        """
        Add item information to the shopping list items, including price and quantity. The default quantity is 1
        :param item: string, Item to be added
        :param price: float, The price of the item
        :param quantity: int, The number of items, defaults to 1
        :return: None
        """
        if item in self.items:
            self.items[item]['quantity'] += quantity
        else:
            self.items[item] = {'price': price, 'quantity': quantity}

    def remove_item(self, item, quantity=1):
        """
        Subtract the specified quantity of item from the shopping list items
        :param item: string, Item to be subtracted in quantity
        :param quantity: int, Quantity to be subtracted
        :return: None
        """
        if item in self.items and self.items[item]['quantity'] >= quantity:
            self.items[item]['quantity'] -= quantity
            if self.items[item]['quantity'] == 0:
                del self.items[item]

    def view_items(self) -> dict:
        """
        Return the current shopping list items
        :return: dict, the current shopping list items
        """
        return self.items

    def total_price(self) -> float:
        """
        Calculate the total price of all items in the shopping list, which is the quantity of each item multiplied by the price
        :return: float, the total price of all items in the shopping list
        """
        total = 0
        for item in self.items:
            total += self.items[item]['price'] * self.items[item]['quantity']
        return total


if __name__ == "__main__":
    shoppingcart = ShoppingCart()

    # Test add_item
    shoppingcart.add_item("apple", 1, 5)
    print(shoppingcart.view_items())  # Output: {'apple': {'price': 1, 'quantity': 5}}

    # Test remove_item
    shoppingcart.remove_item("apple", 3)
    print(shoppingcart.view_items())  # Output: {'apple': {'price': 1, 'quantity': 2}}

    # Test view_items
    shoppingcart.add_item("banana", 2, 3)
    print(shoppingcart.view_items())  # Output: {'apple': {'price': 1, 'quantity': 2}, 'banana': {'price': 2, 'quantity': 3}}

    # Test total_price
    print(shoppingcart.total_price())  # Output: 7.0