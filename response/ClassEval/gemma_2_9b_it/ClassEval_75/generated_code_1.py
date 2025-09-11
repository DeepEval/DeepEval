class ShoppingCart:
    """
    The class manages items, their prices, quantities, and allows to for add, removie, view items, and calculate the total price.
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
        >>> shoppingcart = ShoppingCart()
        >>> shoppingcart.add_item("apple", 1, 5)
        self.items = {"apple":{"price":1, "quantity":5}}
        """
        self.items[item] = {"price": price, "quantity": quantity}

    def remove_item(self, item, quantity=1):
        """
        Subtract the specified quantity of item from the shopping list items
        :param item:string, Item to be subtracted in quantity
        :param quantity:int, Quantity to be subtracted
        :return:None
        >>> shoppingcart.add_item("apple", 1, 5)
        >>> shoppingcart.remove_item("apple", 3)
        self.items = {"apple":{"price":1, "quantity":2}}
        """
        if item in self.items:
            if self.items[item]["quantity"] >= quantity:
                self.items[item]["quantity"] -= quantity
            else:
                print("Not enough items in stock.")
        else:
            print(f"{item} is not in the shopping cart.")

    def view_items(self) -> dict:
        """
        Return the current shopping list items
        :return:dict, the current shopping list items
        >>> shoppingcart.add_item("apple", 1, 5)
        >>> shoppingcart.remove_item("apple", 3)
        >>> shoppingcart.view_items()
        {"apple":{"price":1, "quantity":2}}
        """
        return self.items

    def total_price(self) -> float:
        """
        Calculate the total price of all items in the shopping list, which is the quantity of each item multiplied by the price
        :return:float, the total price of all items in the shopping list
        >>> shoppingcart = ShoppingCart()
        >>> shoppingcart.add_item("apple", 1, 5)
        >>> shoppingcart.add_item("banana", 2, 3)
        >>> shoppingcart.total_price()
        11.0
        """
        total = 0
        for item, details in self.items.items():
            total += details["price"] * details["quantity"]
        return total

if __name__ == "__main__":
    shoppingcart = ShoppingCart()

    # Test case for add_item
    shoppingcart.add_item("apple", 1, 5)
    print(shoppingcart.view_items())  

    # Test case for remove_item
    shoppingcart.remove_item("apple", 3)
    print(shoppingcart.view_items())

    # Test case for view_items
    print(shoppingcart.view_items())

    # Test case for total_price
    shoppingcart.add_item("banana", 2, 3)
    print(shoppingcart.total_price())