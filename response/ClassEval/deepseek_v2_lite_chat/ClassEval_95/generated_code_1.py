class Warehouse:
    """
    The class manages inventory and orders, including adding products, updating product quantities, retrieving product quantities, creating orders, changing order statuses, and tracking orders.
    """

    def __init__(self):
        """
        Initialize two fields.
        self.inventory is a dict that stores the products.
        self.inventory = {Product ID: Product}
        self.orders is a dict that stores the products in a order.
        self.orders = {Order ID: Order}
        """
        self.inventory = {}  # Product ID: Product
        self.orders = {}  # Order ID: Order

    def add_product(self, product_id, name, quantity):
        """
        Add product to inventory and plus the quantity if it has existed in inventory.
        Or just add new product to dict otherwise.
        :param product_id: int
        :param name: str, product name
        :param quantity: int, product quantity
        """
        if product_id in self.inventory:
            self.inventory[product_id].quantity += quantity
            print(self.inventory[product_id])
        else:
            self.inventory[product_id] = {'name': name, 'quantity': quantity}
            print(self.inventory[product_id])

    def update_product_quantity(self, product_id, quantity):
        """
        According to product_id, add the quantity to the corresponding product in inventory.
        """
        if product_id in self.inventory:
            self.inventory[product_id].quantity += quantity
            print(self.inventory[product_id])
        else:
            print(False)

    def get_product_quantity(self, product_id):
        """
        Get the quantity of specific product by product_id.
        :param product_id, int
        :return: if the product_id is in inventory then return the corresponding quantity,
                or False otherwise.
        """
        if product_id in self.inventory:
            print(self.inventory[product_id]['quantity'])
        else:
            print(False)

    def create_order(self, order_id, product_id, quantity):
        """
        Create a order which includes the infomation of product, like id and quantity.
        And put the new order into self.orders.
        The default value of status is 'Shipped'.
        :param order_id: int
        :param product_id: int
        :param quantity: the quantity of product that be selected.
        """
        if product_id in self.inventory:
            if self.inventory[product_id]['quantity'] >= quantity:
                self.orders[order_id] = {'product_id': product_id, 'quantity': quantity, 'status': 'Shipped'}
                self.inventory[product_id]['quantity'] -= quantity
                print(self.orders)
            else:
                print(False)
        else:
            print(False)

    def change_order_status(self, order_id, status):
        """
        Change the status of order if the input order_id is in self.orders.
        :param order_id: int
        :param status: str, the state that is going to change to
        """
        if order_id in self.orders:
            self.orders[order_id]['status'] = status
            print(self.orders)
        else:
            print(False)

    def track_order(self, order_id):
        """
        Get the status of specific order.
        :param order_id: int
        :return False: only if the order_id is not in self.orders.
        """
        if order_id in self.orders:
            print(self.orders[order_id]['status'])
        else:
            print(False)


# Test cases
if __name__ == "__main__":
    warehouse = Warehouse()

    # Test case for add_product
    warehouse.add_product(1, "product1", 3)
    warehouse.add_product(1, "product1", -1)

    # Test case for update_product_quantity
    warehouse.add_product(1, "product1", 3)
    warehouse.update_product_quantity(1, -1)

    # Test case for get_product_quantity
    warehouse.add_product(1, "product1", 3)
    warehouse.get_product_quantity(1)
    warehouse.get_product_quantity(2)

    # Test case for create_order
    warehouse.add_product(1, "product1", 3)
    warehouse.create_order(1, 1, 2)
    warehouse.create_order(1, 2, 2)

    # Test case for change_order_status
    warehouse.add_product(1, "product1", 3)
    warehouse.create_order(1, 1, 2)
    warehouse.change_order_status(1, "done")

    # Test case for track_order
    warehouse.add_product(1, "product1", 3)
    warehouse.create_order(1, 1, 2)
    warehouse.track_order(1)