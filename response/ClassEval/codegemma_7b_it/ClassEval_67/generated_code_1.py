class Order:
    """
    The class manages restaurant orders by allowing the addition of dishes, calculation of the total cost, and checkout.
    """


    def __init__(self):
        """
        Initialize the order management system
        self.menu stores the dishes of resturant inventory
        menu = [{"dish": dish name, "price": price, "count": count}, ...]
        self.selected_dishes stores the dished selected by customer
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
        >>> order = Order()
        >>> order.menu.append({"dish": "dish1", "price": 10, "count": 5})
        >>> order.add_dish({"dish": "dish1", "price": 10, "count": 3})
        True
        """
        if dish["count"] > self.menu[self.menu.index(next((item for item in self.menu if item["dish"] == dish["dish"]), None))]["count"]:
            return False
        self.selected_dishes.append(dish)
        self.menu[self.menu.index(next((item for item in self.menu if item["dish"] == dish["dish"]), None))]["count"] -= dish["count"]
        return True


    def calculate_total(self):
        """
        Calculate the total price of dishes that have been ordered. Multiply the count, price and sales.
        :return total: float, the final total price.
        >>> order = Order()
        >>> order.menu.append({"dish": "dish1", "price": 10, "count": 5})
        >>> order.sales = {"dish1": 0.8}
        >>> order.add_dish({"dish": "dish1", "price": 10, "count": 4})
        True
        >>> order.calculate_total()
        32.0
        """
        total = 0
        for dish in self.selected_dishes:
            total += dish["count"] * dish["price"] * self.sales.get(dish["dish"], 1)
        return total


    def checkout(self):
        """
        Check out the dished ordered. IF the self.selected_dishes is not empty, invoke the calculate_total
        method to check out.
        :return Flase if the self.selected_dishes is empty, or total(return value of calculate_total) otherwise.
        >>> order = Order()
        >>> order.menu.append({"dish": "dish1", "price": 10, "count": 5})
        >>> order.sales = {"dish1": 0.8}
        >>> order.add_dish({"dish": "dish1", "price": 10, "count": 4})
        True
        >>> order.checkout()
        32.0
        """
        if not self.selected_dishes:
            return False
        return self.calculate_total()


if __name__ == "__main__":
    order = Order()
    order.menu.append({"dish": "dish1", "price": 10, "count": 5})
    order.menu.append({"dish": "dish2", "price": 15, "count": 10})
    order.menu.append({"dish": "dish3", "price": 20, "count": 15})
    order.sales = {"dish1": 0.8, "dish2": 0.9, "dish3": 1.0}
    order.add_dish({"dish": "dish1", "price": 10, "count": 3})
    order.add_dish({"dish": "dish2", "price": 15, "count": 5})
    order.add_dish({"dish": "dish3", "price": 20, "count": 10})
    print(order.selected_dishes)
    print(order.calculate_total())
    print(order.checkout())