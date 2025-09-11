import json

class ShoppingCart:
    def __init__(self):
        self.items = {}

    def add_item(self, item, price, quantity=1):
        self.items[item] = {'price': price, 'quantity': quantity}

    def remove_item(self, item, quantity=1):
        if item not in self.items:
            return
        self.items[item]['quantity'] -= quantity
        if self.items[item]['quantity'] <= 0:
            del self.items[item]

    def view_items(self) -> dict:
        return self.items

    def total_price(self) -> float:
        total = 0.0
        for item, data in self.items.items():
            total += data['quantity'] * data['price']
        return total

if __name__ == "__main__":
    shopping_cart = ShoppingCart()
    shopping_cart.add_item("apple", 1, 5)
    shopping_cart.add_item("banana", 2, 3)
    print(shopping_cart.view_items())
    print(shopping_cart.total_price())