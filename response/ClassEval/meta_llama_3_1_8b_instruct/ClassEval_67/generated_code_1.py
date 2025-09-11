class Order:
    """
    The class manages restaurant orders by allowing the addition of dishes, calculation
    of the total cost, and checkout.
    """

    def __init__(self):
        """
        Initialize the order management system
        self.menu stores the dishes of restaurant inventory
        menu = [{"dish": dish name, "price": price, "count": count},...]
        self.selected_dishes stores the dishes selected by customer
        selected_dish = {"dish": dish name, "count": count, "price": price}
        self.sales stores the sales of each dish
        sales = {dish name: sales}
        """
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        """
        Check the self.menu and add into self.selected_dish if the dish count is
        valid.
        And if the dish has successfully been added, change the count in self.menu.
        :param dish: dict, the information of dish. dish = {"dish": dish name,
        "count": count, "price": price}
        :return: True if successfully added, or False otherwise.
        """
        for menu_dish in self.menu:
            if menu_dish["dish"] == dish["dish"]:
                if menu_dish["count"] >= dish["count"]:
                    self.selected_dishes.append({"dish": dish["dish"], "count": dish["count"], "price": menu_dish["price"] * dish["count"]})
                    menu_dish["count"] -= dish["count"]
                    return True
        return False

    def calculate_total(self):
        """
        Calculate the total price of dishes that have been ordered. Multiply the
        count, price and sales.
        :return total: float, the final total price.
        """
        total = 0
        for selected_dish in self.selected_dishes:
            total += selected_dish["price"]
        return total

    def checkout(self):
        """
        Check out the dishes ordered. IF the self.selected_dishes is not empty,
        invoke the calculate_total
        method to check out.
        :return: False if the self.selected_dishes is empty, or total(return value
        of calculate_total) otherwise.
        """
        if self.selected_dishes:
            return self.calculate_total()
        return False

if __name__ == "__main__":
    order = Order()
    
    # Test case for add_dish method
    order.menu.append({"dish": "dish1", "price": 10, "count": 5})
    print(order.add_dish({"dish": "dish1", "count": 3, "price": 10}))  # Expected output: True
    print(order.add_dish({"dish": "dish2", "count": 3, "price": 20}))  # Expected output: False
    
    # Test case for calculate_total method
    order.sales = {"dish1": 0.8}
    print(order.calculate_total())  # Expected output: 24.0
    
    # Test case for checkout method
    order.selected_dishes = [{"dish": "dish1", "count": 4, "price": 24}, {"dish": "dish2", "count": 3, "price": 60}]
    print(order.checkout())  # Expected output: 84.0
    
    # Test case for checkout method when self.selected_dishes is empty
    order.selected_dishes = []
    print(order.checkout())  # Expected output: False