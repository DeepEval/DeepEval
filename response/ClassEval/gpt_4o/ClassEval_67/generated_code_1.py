class Order:
    """
    The class manages restaurant orders by allowing the addition of dishes, calculation of the total cost, and checkout.
    """

    def __init__(self):
        """
        Initialize the order management system
        self.menu stores the dishes of restaurant inventory
        menu = [{"dish": dish name, "price": price, "count": count}, ...]
        self.selected_dishes stores the dishes selected by customer
        selected_dish = {"dish": dish name, "count": count, price: price}
        self.sales stores the sales of each dish
        sales = {dish name: sales}
        """
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        """
        Check the self.menu and add into self.selected_dish if the dish count is valid.
        And if the dish has successfully been added, change the count in self.menu.
        :param dish: dict, the information of dish. dish = {"dish": dish name, "count": count, price: price}
        :return: True if successfully added, or False otherwise.
        """
        for item in self.menu:
            if item["dish"] == dish["dish"] and item["count"] >= dish["count"]:
                self.selected_dishes.append(dish)
                item["count"] -= dish["count"]
                return True
        return False

    def calculate_total(self):
        """
        Calculate the total price of dishes that have been ordered. Multiply the count, price and sales.
        :return total: float, the final total price.
        """
        total = 0.0
        for dish in self.selected_dishes:
            dish_name = dish["dish"]
            count = dish["count"]
            price = dish["price"]
            discount = self.sales.get(dish_name, 1)
            total += count * price * discount
        return total

    def checkout(self):
        """
        Check out the dishes ordered. If the self.selected_dishes is not empty, invoke the calculate_total
        method to check out.
        :return False if the self.selected_dishes is empty, or total(return value of calculate_total) otherwise.
        """
        if not self.selected_dishes:
            return False
        return self.calculate_total()

if __name__ == "__main__":
    # Test case for add_dish method
    order = Order()
    order.menu.append({"dish": "dish1", "price": 10, "count": 5})
    output = order.add_dish({"dish": "dish1", "price": 10, "count": 3})
    print(output)  # Expected: True

    # Test case for calculate_total method
    order.selected_dishes = []  # Reset selected_dishes for a new test
    order.add_dish({"dish": "dish1", "price": 10, "count": 4})
    order.sales = {"dish1": 0.8}
    output = order.calculate_total()
    print(output)  # Expected: 32.0

    # Test case for checkout method
    order.selected_dishes = []  # Reset selected_dishes for a new test
    order.add_dish({"dish": "dish1", "price": 10, "count": 4})
    output = order.checkout()
    print(output)  # Expected: 32.0