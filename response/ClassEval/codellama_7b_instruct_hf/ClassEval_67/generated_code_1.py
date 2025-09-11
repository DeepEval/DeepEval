import json

class Order:
    def __init__(self):
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        if not dish:
            return False
        if dish['count'] > 0:
            self.selected_dishes.append(dish)
            self.menu.append(dish)
            return True
        else:
            return False

    def calculate_total(self):
        total = 0
        for dish in self.selected_dishes:
            total += dish['count'] * dish['price'] * self.sales[dish['dish']]
        return total

    def checkout(self):
        if not self.selected_dishes:
            return False
        total = self.calculate_total()
        return total

if __name__ == "__main__":
    order = Order()
    order.menu.append({"dish": "dish1", "price": 10, "count": 5})
    order.sales = {"dish1": 0.8}
    order.add_dish({"dish": "dish1", "price": 10, "count": 4})
    total = order.checkout()
    print(total)