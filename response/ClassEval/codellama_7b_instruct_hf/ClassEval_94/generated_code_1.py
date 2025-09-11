import collections

class VendingMachine:
    def __init__(self):
        self.inventory = collections.defaultdict(dict)
        self.balance = 0

    def add_item(self, item_name, price, quantity):
        self.inventory[item_name] = {"price": price, "quantity": quantity}

    def insert_coin(self, amount):
        self.balance += amount
        return self.balance

    def purchase_item(self, item_name):
        if self.balance < self.inventory[item_name]["price"]:
            return False
        else:
            self.balance -= self.inventory[item_name]["price"]
            self.inventory[item_name]["quantity"] -= 1
            return self.balance

    def restock_item(self, item_name, quantity):
        if item_name not in self.inventory:
            return False
        else:
            self.inventory[item_name]["quantity"] += quantity
            return True

    def display_items(self):
        items = []
        for item_name, item in self.inventory.items():
            items.append(f"{item_name} - ${item['price']} [{item['quantity']}]")
        if not items:
            return False
        else:
            return "\n".join(items)
        
if __name__ == "__main__":
    machine = VendingMachine()
    machine.add_item("Soda", 1.50, 10)
    machine.add_item("Chips", 1.00, 5)
    print(machine.display_items())
    
    print("Inserting $2.00")
    print("Balance:", machine.insert_coin(2.00))
    
    print("Purchasing Soda")
    print("Remaining Balance:", machine.purchase_item("Soda"))
    
    print("Restocking Chips")
    print(machine.restock_item("Chips", 10))
    
    print(machine.display_items())