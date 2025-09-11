class VendingMachine:
    """
    This is a class to simulate a vending machine, including adding products, inserting coins, purchasing products, viewing balance, replenishing product inventory, and displaying product information.
    """

    def __init__(self):
        """
        Initializes the vending machine's inventory and balance.
        """
        self.inventory = {}
        self.balance = 0

    def add_item(self, item_name, price, quantity):
        """
        Adds a product to the vending machine's inventory.
        :param item_name: The name of the product to be added, str.
        :param price: The price of the product to be added, float.
        :param quantity: The quantity of the product to be added, int.
        :return: None
        >>> vendingMachine = VendingMachine()
        >>> vendingMachine.add_item('Coke', 1.25, 10)
        >>> vendingMachine.inventory
        {'Coke': {'price': 1.25, 'quantity': 10}}

        """
        if item_name in self.inventory:
            self.inventory[item_name]['quantity'] += quantity
        else:
            self.inventory[item_name] = {'price': price, 'quantity': quantity}

    def insert_coin(self, amount):
        """
        Inserts coins into the vending machine.
        :param amount: The amount of coins to be inserted, float.
        :return: The balance of the vending machine after the coins are inserted, float.
        >>> vendingMachine = VendingMachine()
        >>> vendingMachine.insert_coin(1.25)
        1.25

        """
        self.balance += amount
        return self.balance

    def purchase_item(self, item_name):
        """
        Purchases a product from the vending machine and returns the balance after the purchase and display purchase unsuccessful if the product is out of stock.
        :param item_name: The name of the product to be purchased, str.
        :return: If successful, returns the balance of the vending machine after the product is purchased, float,otherwise,returns False.
        >>> vendingMachine = VendingMachine()
        >>> vendingMachine.inventory = {'Coke': {'price': 1.25, 'quantity': 10}}
        >>> vendingMachine.balance = 1.25
        >>> vendingMachine.purchase_item('Coke')
        0.0
        >>> vendingMachine.purchase_item('Pizza')
        False

        """
        if item_name in self.inventory and self.inventory[item_name]['quantity'] > 0 and self.balance >= self.inventory[item_name]['price']:
            self.balance -= self.inventory[item_name]['price']
            self.inventory[item_name]['quantity'] -= 1
            return self.balance
        else:
            return False

    def restock_item(self, item_name, quantity):
        """
        Replenishes the inventory of a product already in the vending machine.
        :param item_name: The name of the product to be replenished, str.
        :param quantity: The quantity of the product to be replenished, int.
        :return: If the product is already in the vending machine, returns True, otherwise, returns False.
        >>> vendingMachine = VendingMachine()
        >>> vendingMachine.inventory = {'Coke': {'price': 1.25, 'quantity': 10}}
        >>> vendingMachine.restock_item('Coke', 10)
        True
        >>> vendingMachine.restock_item('Pizza', 10)
        False

        """
        if item_name in self.inventory:
            self.inventory[item_name]['quantity'] += quantity
            return True
        else:
            return False

    def display_items(self):
        """
        Displays the products in the vending machine.
        :return: If the vending machine is empty, returns False, otherwise, returns a list of the products in the vending machine, str.
        >>> vendingMachine = VendingMachine()
        >>> vendingMachine.display_items()
        False
        >>> vendingMachine.inventory = {'Coke': {'price': 1.25, 'quantity': 10} }
        >>> vendingMachine.display_items()
        'Coke - $1.25 [10]'

        """
        if not self.inventory:
            return False
        else:
            items = []
            for item, details in self.inventory.items():
                items.append(f"{item} - ${details['price']} [{details['quantity']}]")
            return "\n".join(items)

if __name__ == "__main__":
    vendingMachine = VendingMachine()
    vendingMachine.add_item('Coke', 1.25, 10)
    vendingMachine.add_item('Chips', 0.75, 5)
    vendingMachine.add_item('Candy', 0.50, 10)
    
    print(vendingMachine.display_items())
    
    print(vendingMachine.insert_coin(1.25))
    
    print(vendingMachine.purchase_item('Coke'))
    
    print(vendingMachine.purchase_item('Pizza'))
    
    print(vendingMachine.restock_item('Coke', 10))
    
    print(vendingMachine.display_items())