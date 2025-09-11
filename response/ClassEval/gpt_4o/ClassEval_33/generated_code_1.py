class DiscountStrategy:
    """
    This is a class that allows using different discount strategies based on shopping credit or shopping cart in a supermarket.
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
        return sum(item['quantity'] * item['price'] for item in self.cart)

    def due(self):
        """
        Calculate the final amount to be paid after applying the discount.
        :return: float, final amount to be paid
        """
        if self.promotion:
            discount = self.promotion(self)
        else:
            discount = 0
        return self.total() - discount

    @staticmethod
    def FidelityPromo(order):
        """
        Calculate the discount based on the fidelity points of the customer.
        Customers with over 1000 points can enjoy a 5% discount on the entire order.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        """
        if order.customer['fidelity'] > 1000:
            return order.total() * 0.05
        return 0.0

    @staticmethod
    def BulkItemPromo(order):
        """
        Calculate the discount based on bulk item quantity in the order.
        In the same order, if the quantity of a single item reaches 20 or more, each item will enjoy a 10% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        """
        discount = 0
        for item in order.cart:
            if item['quantity'] >= 20:
                discount += item['quantity'] * item['price'] * 0.10
        return discount

    @staticmethod
    def LargeOrderPromo(order):
        """
        Calculate the discount based on the number of different products in the order.
        If the quantity of different products in the order reaches 10 or more, the entire order will enjoy a 7% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        """
        if len(order.cart) >= 10:
            return order.total() * 0.07
        return 0.0

if __name__ == "__main__":
    # Test case for total method
    customer = {'name': 'John Doe', 'fidelity': 1200}
    cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
    ds = DiscountStrategy(customer, cart)
    print(ds.total())  # Expected output: 329.0

    # Test case for due method with FidelityPromo
    ds = DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)
    print(ds.due())  # Expected output: 312.55

    # Test case for FidelityPromo method
    print(DiscountStrategy.FidelityPromo(ds))  # Expected output: 16.45

    # Test case for BulkItemPromo method
    cart_bulk = [{'product': 'product', 'quantity': 20, 'price': 23.5}]
    ds_bulk = DiscountStrategy(customer, cart_bulk, DiscountStrategy.BulkItemPromo)
    print(DiscountStrategy.BulkItemPromo(ds_bulk))  # Expected output: 47.0

    # Test case for LargeOrderPromo method
    cart_large = [{'product': f'product{i}', 'quantity': 1, 'price': 10.0} for i in range(10)]
    ds_large = DiscountStrategy(customer, cart_large, DiscountStrategy.LargeOrderPromo)
    print(DiscountStrategy.LargeOrderPromo(ds_large))  # Expected output: 7.0