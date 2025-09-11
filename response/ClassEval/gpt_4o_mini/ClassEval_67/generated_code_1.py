class Order:
    """
    The class manages restaurant orders by allowing the addition of dishes, calculation of the total cost, and checkout.
    """

    def __init__(self):
        """
        Initialize the order management system.
        self.menu stores the dishes of restaurant inventory.
        menu = [{"dish": dish name, "price": price, "count": count}, ...]
        self.selected_dishes stores the dishes selected by customer.
        selected_dishes = {"dish": dish name, "count": count, price: price}
        self.sales stores the sales of each dish.
        sales = {dish name: sales}
        """
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        """
        Check the self.menu and add into self.selected_dishes if the dish count is valid.
        And if the dish has successfully been added, change the count in self.menu.
        :param dish: dict, the information of dish. dish = {"dish": dish name, "count": count, price: price}
        :return: True if successfully added, or False otherwise.
        """
        dish_name = dish["dish"]
        dish_count = dish["count"]

        # Check if the dish is in the menu
        for menu_item in self.menu:
            if menu_item["dish"] == dish_name:
                if menu_item["count"] >= dish_count:
                    # Update menu count
                    menu_item["count"] -= dish_count
                    # Add to selected_dishes
                    self.selected_dishes.append(dish)
                    return True
                else:
                    return False
        return False

    def calculate_total(self):
        """
        Calculate the total price of dishes that have been ordered. Multiply the count, price and sales.
        :return total: float, the final total price.
        """
        total = 0.0
        for selected in self.selected_dishes:
            dish_name = selected["dish"]
            count = selected["count"]
            price = selected["price"]
            discount = self.sales.get(dish_name, 1)  # Default discount is 1 (no discount)
            total += count * price * discount
        return total

    def checkout(self):
        """
        Check out the dishes ordered. IF the self.selected_dishes is not empty, invoke the calculate_total
        method to check out.
        :return: False if the self.selected_dishes is empty, or total (return value of calculate_total) otherwise.
        """
        if not self.selected_dishes:
            return False
        return self.calculate_total()


# Test cases
if __name__ == "__main__":
    order = Order()
    order.menu.append({"dish": "dish1", "price": 10, "count": 5})
    print(order.add_dish({"dish": "dish1", "price": 10, "count": 3}))  # True
    print(order.menu)  # [{"dish": "dish1", "price": 10, "count": 2}]
    
    order.sales = {"dish1": 0.8}
    print(order.add_dish({"dish": "dish1", "price": 10, "count": 4}))  # False (not enough count)
    
    print(order.calculate_total())  # 24.0 (3 dishes of dish1 with discount)
    
    print(order.checkout())  # 24.0 (total price)
    
    print(order.checkout())  # 24.0 again (checkout not empty)
    
    # Testing checkout with no selected dishes
    order2 = Order()
    print(order2.checkout())  # False (no dishes selected)