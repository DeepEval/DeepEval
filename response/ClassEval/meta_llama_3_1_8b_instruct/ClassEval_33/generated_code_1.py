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
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)

        """
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.total()

    def total(self):
        """
        Calculate the total cost of items in the cart.
        :return: float, total cost of items
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> ds = DiscountStrategy(customer, cart)
        >>> ds.total()
        329.0

        """
        self.total_cost = sum(item['quantity'] * item['price'] for item in self.cart)
        return self.total_cost

    def due(self):
        """
        Calculate the final amount to be paid after applying the discount.
        :return: float, final amount to be paid
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> ds = DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)
        >>> ds.due()
        312.55

        """
        if self.promotion:
            return self.total_cost - self.promotion(self)
        else:
            return self.total_cost

    @staticmethod
    def FidelityPromo(order):
        """
        Calculate the discount based on the fidelity points of the customer.Customers with over 1000 points can enjoy a 5% discount on the entire order.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> order = DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)
        >>> DiscountStrategy.FidelityPromo(order)
        16.45

        """
        if order.customer['fidelity'] > 1000:
            return order.total_cost * 0.05
        else:
            return 0

    @staticmethod
    def BulkItemPromo(order):
        """
        Calculate the discount based on bulk item quantity in the order.In the same order, if the quantity of a single item reaches 20 or more, each item will enjoy a 10% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 20, 'price': 23.5}]
        >>> order = DiscountStrategy(customer, cart, DiscountStrategy.BulkItemPromo)
        >>> DiscountStrategy.BulkItemPromo(order)
        47.0

        """
        discount = 0
        for item in order.cart:
            if item['quantity'] >= 20:
                discount += item['quantity'] * item['price'] * 0.10
        return discount

    @staticmethod
    def LargeOrderPromo(order):
        """
        Calculate the discount based on the number of different products in the order.If the quantity of different products in the order reaches 10 or more, the entire order will enjoy a 7% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> order = DiscountStrategy(customer, cart, DiscountStrategy.LargeOrderPromo)
        >>> DiscountStrategy.LargeOrderPromo(order)
        0.0

        """
        if len(set(item['product'] for item in order.cart)) >= 10:
            return order.total_cost * 0.07
        else:
            return 0

if __name__ == "__main__":
    customer = {'name': 'John Doe', 'fidelity': 1200}
    cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
    ds = DiscountStrategy(customer, cart)
    print("Total cost:", ds.total())

    ds_with_promo = DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)
    print("Due with FidelityPromo:", ds_with_promo.due())

    ds_with_bulk_promo = DiscountStrategy(customer, cart, DiscountStrategy.BulkItemPromo)
    print("Due with BulkItemPromo:", ds_with_bulk_promo.due())

    ds_with_large_promo = DiscountStrategy(customer, cart, DiscountStrategy.LargeOrderPromo)
    print("Due with LargeOrderPromo:", ds_with_large_promo.due())

    customer_with_bulk = {'name': 'John Doe', 'fidelity': 1200}
    cart_with_bulk = [{'product': 'product1', 'quantity': 20, 'price': 23.5}, 
                      {'product': 'product2', 'quantity': 15, 'price': 25.0}]
    ds_with_bulk = DiscountStrategy(customer_with_bulk, cart_with_bulk, DiscountStrategy.BulkItemPromo)
    print("Due with BulkItemPromo for bulk items:", ds_with_bulk.due())

    customer_with_large = {'name': 'John Doe', 'fidelity': 1200}
    cart_with_large = [{'product': 'product1', 'quantity': 12, 'price': 23.5}, 
                       {'product': 'product2', 'quantity': 12, 'price': 25.0}, 
                       {'product': 'product3', 'quantity': 12, 'price': 27.5}, 
                       {'product': 'product4', 'quantity': 12, 'price': 30.0}, 
                       {'product': 'product5', 'quantity': 12, 'price': 32.5}, 
                       {'product': 'product6', 'quantity': 12, 'price': 35.0}, 
                       {'product': 'product7', 'quantity': 12, 'price': 37.5}, 
                       {'product': 'product8', 'quantity': 12, 'price': 40.0}, 
                       {'product': 'product9', 'quantity': 12, 'price': 42.5}, 
                       {'product': 'product10', 'quantity': 12, 'price': 45.0}]
    ds_with_large = DiscountStrategy(customer_with_large, cart_with_large, DiscountStrategy.LargeOrderPromo)
    print("Due with LargeOrderPromo for large order:", ds_with_large.due())