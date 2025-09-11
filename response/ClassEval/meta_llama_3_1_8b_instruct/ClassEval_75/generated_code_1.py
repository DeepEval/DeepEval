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
        :param quantity:int, The number of items, defaults to 1
        :return:None
        """
        if item not in self.items:
            self.items[item] = {"price": price, "quantity": quantity}
        else:
            self.items[item]["price"] = price
            self.items[item]["quantity"] += quantity

    def remove_item(self, item, quantity=1):
        """
        Subtract the specified quantity of item from the shopping list items
        :param item:string, Item to be subtracted in quantity
        :param quantity:int, Quantity to be subtracted
        :return:None
        """
        if item in self.items:
            if self.items[item]["quantity"] <= quantity:
                del self.items[item]
            else:
                self.items[item]["quantity"] -= quantity

    def view_items(self) -> dict:
        """
        Return the current shopping list items
        :return:dict, the current shopping list items
        """
        return self.items.copy()

    def total_price(self) -> float:
        """
        Calculate the total price of all items in the shopping list, which is the quantity of each item multiplied by the price
        :return:float, the total price of all items in the shopping list
        """
        total_price = 0
        for item in self.items:
            total_price += self.items[item]["quantity"] * self.items[item]["price"]
        return total_price


if __name__ == "__main__":
    shoppingcart = ShoppingCart()
    
    # Test case for add_item
    shoppingcart.add_item("apple", 1, 5)
    print("Test case for add_item: ", shoppingcart.view_items())  # Expected output: {"apple": {"price": 1, "quantity": 5}}
    
    # Test case for remove_item
    shoppingcart.remove_item("apple", 3)
    print("Test case for remove_item: ", shoppingcart.view_items())  # Expected output: {"apple": {"price": 1, "quantity": 2}}
    
    # Test case for view_items
    print("Test case for view_items: ", shoppingcart.view_items())  # Expected output: {"apple": {"price": 1, "quantity": 2}}
    
    # Test case for total_price
    shoppingcart.add_item("banana", 2, 3)
    print("Test case for total_price: ", shoppingcart.total_price())  # Expected output: 11.0
    
    # Test case for total_price with no items
    print("Test case for total_price with no items: ", shoppingcart.total_price())  # Expected output: 0.0
    
    # Test case for add_item with existing item
    shoppingcart.add_item("apple", 2, 5)
    print("Test case for add_item with existing item: ", shoppingcart.view_items())  # Expected output: {"apple": {"price": 2, "quantity": 10}}