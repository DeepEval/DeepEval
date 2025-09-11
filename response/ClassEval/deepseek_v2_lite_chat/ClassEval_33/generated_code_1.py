import math

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
        self.total()

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
        total_cost = self.total()
        if self.promotion is None:
            return total_cost
        else:
            return self.promotion(self)

    @staticmethod
    def FidelityPromo(order):
        """
        Calculate the discount based on the fidelity points of the customer.
        Customers with over 1000 points can enjoy a 5% discount on the entire order.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        """
        discount = (order['total'] * 5) / 100
        return discount

    @staticmethod
    def BulkItemPromo(order):
        """
        Calculate the discount based on bulk item quantity in the order.
        In the same order, if the quantity of a single item reaches 20 or more, each item will enjoy a 10% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        """
        total_cost = order['total']
        if total_cost > 200:
            discount = (total_cost * 10) / 100
        else:
            discount = 0
        return discount

    @staticmethod
    def LargeOrderPromo(order):
        """
        Calculate the discount based on the number of different products in the order.
        If the quantity of different products in the order reaches 10 or more, the entire order will enjoy a 7% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        """
        total_cost = order['total']
        num_different_products = len(set(item['product'] for item in order['cart']))
        if num_different_products > 10:
            discount = (total_cost * 7) / 100
        else:
            discount = 0
        return discount

# Test cases
if __name__ == "__main__":
    # Test Case 1: Fidelity discount
    customer = {'name': 'John Doe', 'fidelity': 1200}
    cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
    ds = DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)
    print("Total cost without discount:", ds.total())
    print("Discount:", ds.due())
    # Test Case 2: Bulk item discount
    ds = DiscountStrategy(customer, cart, DiscountStrategy.BulkItemPromo)
    print("Total cost without discount:", ds.total())
    print("Discount:", ds.due())
    # Test Case 3: Large order discount
    ds = DiscountStrategy(customer, cart, DiscountStrategy.LargeOrderPromo)
    print("Total cost without discount:", ds.total())
    print("Discount:", ds.due())