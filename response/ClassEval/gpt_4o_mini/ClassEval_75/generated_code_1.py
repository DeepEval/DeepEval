class ShoppingCart:
    """
    The class manages items, their prices, quantities, and allows to add, remove, view items, and calculate the total price.
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
        :param quantity:int, The number of items, defaults to 1
        :return: None
        """
        if item in self.items:
            self.items[item]["quantity"] += quantity
        else:
            self.items[item] = {"price": price, "quantity": quantity}

    def remove_item(self, item, quantity=1):
        """
        Subtract the specified quantity of item from the shopping list items
        :param item: string, Item to be subtracted in quantity
        :param quantity: int, Quantity to be subtracted
        :return: None
        """
        if item in self.items:
            if self.items[item]["quantity"] <= quantity:
                del self.items[item]  # remove the item if quantity drops to 0 or less
            else:
                self.items[item]["quantity"] -= quantity

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
        total = sum(item["price"] * item["quantity"] for item in self.items.values())
        return total


if __name__ == "__main__":
    shoppingcart = ShoppingCart()

    # Test add_item
    shoppingcart.add_item("apple", 1, 5)
    print("After adding apples:", shoppingcart.view_items())  # Expected: {"apple": {"price": 1, "quantity": 5}}

    # Test add_item with existing item
    shoppingcart.add_item("apple", 1, 3)
    print("After adding more apples:", shoppingcart.view_items())  # Expected: {"apple": {"price": 1, "quantity": 8}}

    # Test remove_item
    shoppingcart.remove_item("apple", 3)
    print("After removing apples:", shoppingcart.view_items())  # Expected: {"apple": {"price": 1, "quantity": 5}}

    # Test remove_item to zero quantity
    shoppingcart.remove_item("apple", 5)
    print("After removing all apples:", shoppingcart.view_items())  # Expected: {}

    # Test total_price
    shoppingcart.add_item("banana", 2, 3)
    shoppingcart.add_item("orange", 3, 2)
    print("Total price:", shoppingcart.total_price())  # Expected: 12.0 (3*2 + 2*3)

    # Final view of items
    print("Final items in cart:", shoppingcart.view_items())  # Expected: {"banana": {"price": 2, "quantity": 3}, "orange": {"price": 3, "quantity": 2}}