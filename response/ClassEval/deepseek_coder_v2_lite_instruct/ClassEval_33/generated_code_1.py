class DiscountStrategy:
    """
    This is a class that allows to use different discount strategy based on shopping credit or shopping cart in supermarket.
    """

    def __init__(self, customer, cart, promotion=None):
        """
        Initialize the DiscountStrategy with customer information, a cart of items, and an optional promotion.
        :param customer: dict, customer information
        :param cart: list of dicts, a cart of items with details
        :param promotion: function, optional promotion applied to the order
        """
        self.customer = customer
        self.cart = cart
        self.promotion = promotion

    def total(self):
        """
        Calculate the total cost of items in the cart.
        :return: float, total cost of items
        """
        total_cost = sum(item['quantity'] * item['price'] for item in self.cart)
        return total_cost

    def due(self):
        """
        Calculate the final amount to be paid after applying the discount.
        :return: float, final amount to be paid
        """
        if self.promotion is None:
            return self.total()
        else:
            discount = self.promotion(self)
            return self.total() - discount

    @staticmethod
    def FidelityPromo(order):
        """
        Calculate the discount based on the fidelity points of the customer.Customers with over 1000 points can enjoy a 5% discount on the entire order.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        """
        if order.customer['fidelity'] >= 1000:
            total_cost = order.total()
            discount = total_cost * 0.05
            return discount
        return 0

    @staticmethod
    def BulkItemPromo(order):
        """
        Calculate the discount based on bulk item quantity in the order.In the same order, if the quantity of a single item reaches 20 or more, each item will enjoy a 10% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        """
        discount = 0
        for item in order.cart:
            if item['quantity'] >= 20:
                discount += item['quantity'] * item['price'] * 0.1
        return discount

    @staticmethod
    def LargeOrderPromo(order):
        """
        Calculate the discount based on the number of different products in the order.If the quantity of different products in the order reaches 10 or more, the entire order will enjoy a 7% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        """
        distinct_products = {item['product'] for item in order.cart}
        if len(distinct_products) >= 10:
            total_cost = order.total()
            discount = total_cost * 0.07
            return discount
        return 0

# Test cases
if __name__ == "__main__":
    # Test the total method
    customer = {'name': 'John Doe', 'fidelity': 1200}
    cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
    ds = DiscountStrategy(customer, cart)
    print(ds.total())  # Expected output: 329.0

    # Test the due method without promotion
    print(ds.due())  # Expected output: 329.0

    # Test the FidelityPromo method
    ds = DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)
    print(DiscountStrategy.FidelityPromo(ds))  # Expected output: 16.45
    print(ds.due())  # Expected output: 312.55

    # Test the BulkItemPromo method
    cart = [{'product': 'product', 'quantity': 20, 'price': 23.5}]
    ds = DiscountStrategy(customer, cart, DiscountStrategy.BulkItemPromo)
    print(DiscountStrategy.BulkItemPromo(ds))  # Expected output: 47.0
    print(ds.due())  # Expected output: 182.0

    # Test the LargeOrderPromo method
    cart = [{'product': 'product1', 'quantity': 14, 'price': 23.5},
            {'product': 'product2', 'quantity': 14, 'price': 23.5}]
    ds = DiscountStrategy(customer, cart, DiscountStrategy.LargeOrderPromo)
    print(DiscountStrategy.LargeOrderPromo(ds))  # Expected output: 0.0
    print(ds.due())  # Expected output: 658.0