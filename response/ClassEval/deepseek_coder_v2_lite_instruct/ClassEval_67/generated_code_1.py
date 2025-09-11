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
        selected_dish = {"dish": dish name, "count": count, "price": price}
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
        :param dish: dict, the information of dish. dish = {"dish": dish name, "price": price, "count": count}
        :return: True if successfully added, or False otherwise.
        """
        for item in self.menu:
            if item["dish"] == dish["dish"]:
                if item["count"] >= dish["count"]:
                    self.selected_dishes.append({"dish": dish["dish"], "count": dish["count"], "price": item["price"]})
                    item["count"] -= dish["count"]
                    return True
                else:
                    return False
        return False

    def calculate_total(self):
        """
        Calculate the total price of dishes that have been ordered. Multiply the count, price, and sales.
        :return total: float, the final total price.
        """
        total = 0
        for dish in self.selected_dishes:
            total += dish["count"] * dish["price"] * self.sales.get(dish["dish"], 1)
        return total

    def checkout(self):
        """
        Check out the dishes ordered. If the self.selected_dishes is not empty, invoke the calculate_total
        method to check out.
        :return False if the self.selected_dishes is empty, or total (return value of calculate_total) otherwise.
        """
        if not self.selected_dishes:
            return False
        return self.calculate_total()


if __name__ == "__main__":
    # Test cases
    order = Order()
    order.menu.append({"dish": "dish1", "price": 10, "count": 5})
    order.sales = {"dish1": 0.8}
    
    # Test add_dish
    print(order.add_dish({"dish": "dish1", "price": 10, "count": 3}))  # Should return True
    print(order.selected_dishes)  # Should show the added dish
    
    # Test calculate_total
    print(order.calculate_total())  # Should return the calculated total
    
    # Test checkout
    print(order.checkout())  # Should return the total calculated