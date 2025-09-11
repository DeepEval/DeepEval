class VendingMachine:
    def __init__(self):
        self.inventory = {}
        self.balance = 0

    def add_item(self, item_name, price, quantity):
        self.inventory[item_name] = {'price': price, 'quantity': quantity}

    def insert_coin(self, amount):
        self.balance += amount
        return self.balance

    def purchase_item(self, item_name):
        if item_name not in self.inventory or self.inventory[item_name]['quantity'] == 0:
            return False
        cost = self.inventory[item_name]['price']
        self.inventory[item_name]['quantity'] -= 1
        self.balance -= cost
        return self.balance

    def restock_item(self, item_name, quantity):
        if item_name not in self.inventory:
            return False
        self.inventory[item_name]['quantity'] += quantity
        return True

    def display_items(self):
        if not self.inventory:
            return False
        items = []
        for item, details in self.inventory.items():
            items.append(f"{item} - ${details['price']} [{details['quantity']}]")
        return ', '.join(items)

if __name__ == "__main__":
    vm = VendingMachine()
    vm.add_item("Soda", 1.50, 10)
    vm.add_item("Chips", 1.00, 5)
    print(vm.display_items())  # Output: Soda - $1.5 [10], Chips - $1.0 [5]
    vm.insert_coin(5.00)
    print(vm.purchase_item("Soda"))  # Output: 3.5
    vm.restock_item("Chips", 10)
    print(vm.display_items())  # Output: Soda - $1.5 [9], Chips - $1.0 [15]