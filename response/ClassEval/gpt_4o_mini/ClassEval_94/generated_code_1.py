class VendingMachine:
    """
    This is a class to simulate a vending machine, including adding products, 
    inserting coins, purchasing products, viewing balance, replenishing product 
    inventory, and displaying product information.
    """

    def __init__(self):
        """
        Initializes the vending machine's inventory and balance.
        """
        self.inventory = {}
        self.balance = 0.0

    def add_item(self, item_name, price, quantity):
        """
        Adds a product to the vending machine's inventory.
        :param item_name: The name of the product to be added, str.
        :param price: The price of the product to be added, float.
        :param quantity: The quantity of the product to be added, int.
        :return: None
        """
        self.inventory[item_name] = {'price': price, 'quantity': quantity}

    def insert_coin(self, amount):
        """
        Inserts coins into the vending machine.
        :param amount: The amount of coins to be inserted, float.
        :return: The balance of the vending machine after the coins are inserted, float.
        """
        self.balance += amount
        return self.balance

    def purchase_item(self, item_name):
        """
        Purchases a product from the vending machine and returns the balance after the purchase 
        or displays purchase unsuccessful if the product is out of stock.
        :param item_name: The name of the product to be purchased, str.
        :return: If successful, returns the balance of the vending machine after the product is 
        purchased, float; otherwise, returns False.
        """
        if item_name in self.inventory:
            item = self.inventory[item_name]
            if item['quantity'] > 0 and self.balance >= item['price']:
                item['quantity'] -= 1
                self.balance -= item['price']
                return self.balance
            elif item['quantity'] == 0:
                print("Purchase unsuccessful: Product is out of stock.")
                return False
            else:
                print("Purchase unsuccessful: Insufficient funds.")
                return False
        else:
            print("Purchase unsuccessful: Product not found.")
            return False

    def restock_item(self, item_name, quantity):
        """
        Replenishes the inventory of a product already in the vending machine.
        :param item_name: The name of the product to be replenished, str.
        :param quantity: The quantity of the product to be replenished, int.
        :return: If the product is already in the vending machine, returns True; otherwise, 
        returns False.
        """
        if item_name in self.inventory:
            self.inventory[item_name]['quantity'] += quantity
            return True
        return False

    def display_items(self):
        """
        Displays the products in the vending machine.
        :return: If the vending machine is empty, returns False; otherwise, returns a list of 
        the products in the vending machine, str.
        """
        if not self.inventory:
            return False
        return '\n'.join(f"{item} - ${details['price']} [{details['quantity']}]" 
                         for item, details in self.inventory.items())


if __name__ == "__main__":
    # Test cases
    vendingMachine = VendingMachine()

    # Test add_item
    vendingMachine.add_item('Coke', 1.25, 10)
    print(vendingMachine.inventory)  # Expected: {'Coke': {'price': 1.25, 'quantity': 10}}

    # Test insert_coin
    print(vendingMachine.insert_coin(1.25))  # Expected: 1.25

    # Test purchase_item
    vendingMachine.purchase_item('Coke')  # Expected: 0.0
    print(vendingMachine.inventory)  # Expected: {'Coke': {'price': 1.25, 'quantity': 9}}

    # Test unsuccessful purchase (not enough balance)
    print(vendingMachine.purchase_item('Coke'))  # Expected: Insufficient funds, False

    # Test restock_item
    print(vendingMachine.restock_item('Coke', 10))  # Expected: True
    print(vendingMachine.inventory['Coke']['quantity'])  # Expected: 10
    print(vendingMachine.restock_item('Pizza', 10))  # Expected: False

    # Test display_items
    print(vendingMachine.display_items())  # Expected: 'Coke - $1.25 [10]'