class Product:
    def __init__(self, name, quantity):
        self.name = name
        self.quantity = quantity

class Order:
    def __init__(self, product_id, quantity, status='Shipped'):
        self.product_id = product_id
        self.quantity = quantity
        self.status = status

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
        else:
            self.inventory[product_id] = Product(name, quantity)

    def update_product_quantity(self, product_id, quantity):
        """
        According to product_id, add the quantity to the corresponding product in inventory.
        """
        if product_id in self.inventory:
            self.inventory[product_id].quantity += quantity
        else:
            print(f"Product {product_id} does not exist in inventory.")

    def get_product_quantity(self, product_id):
        """
        Get the quantity of specific product by product_id.
        :param product_id, int
        :return: if the product_id is in inventory then return the corresponding quantity,
                or False otherwise.
        """
        if product_id in self.inventory:
            return self.inventory[product_id].quantity
        else:
            return False

    def create_order(self, order_id, product_id, quantity):
        """
        Create a order which includes the infomation of product, like id and quantity.
        And put the new order into self.orders.
        The default value of status is 'Shipped'.
        :param order_id: int
        :param product_id: int
        :param quantity: the quantity of product that be selected.
        :return False: only if product_id is not in inventory or the quantity is not adequate
        """
        if product_id in self.inventory and self.inventory[product_id].quantity >= quantity:
            self.orders[order_id] = Order(product_id, quantity)
            self.inventory[product_id].quantity -= quantity
            return True
        else:
            return False

    def change_order_status(self, order_id, status):
        """
        Change the status of order if the input order_id is in self.orders.
        :param order_id: int
        :param status: str, the state that is going to change to
        :return False: only if the order_id is not in self.orders
        """
        if order_id in self.orders:
            self.orders[order_id].status = status
            return True
        else:
            return False

    def track_order(self, order_id):
        """
        Get the status of specific order.
        :param order_id: int
        :return False: only if the order_id is not in self.orders.
        """
        if order_id in self.orders:
            return self.orders[order_id].status
        else:
            return False

if __name__ == "__main__":
    warehouse = Warehouse()

    # Test case for add_product
    print("Test case for add_product:")
    warehouse.add_product(1, "product1", 3)
    print(warehouse.inventory)  # Expected output: {1: {'name': 'product1', 'quantity': 3}}
    warehouse.add_product(1, "product1", 2)
    print(warehouse.inventory)  # Expected output: {1: {'name': 'product1', 'quantity': 5}}

    # Test case for update_product_quantity
    print("\nTest case for update_product_quantity:")
    warehouse.update_product_quantity(1, -1)
    print(warehouse.inventory)  # Expected output: {1: {'name': 'product1', 'quantity': 4}}

    # Test case for get_product_quantity
    print("\nTest case for get_product_quantity:")
    print(warehouse.get_product_quantity(1))  # Expected output: 4
    print(warehouse.get_product_quantity(2))  # Expected output: False

    # Test case for create_order
    print("\nTest case for create_order:")
    warehouse.add_product(1, "product1", 3)
    print(warehouse.create_order(1, 1, 2))  # Expected output: True
    print(warehouse.orders)  # Expected output: {1: {'product_id': 1, 'quantity': 2,'status': 'Shipped'}}
    print(warehouse.create_order(2, 1, 2))  # Expected output: False

    # Test case for change_order_status
    print("\nTest case for change_order_status:")
    warehouse.change_order_status(1, "done")
    print(warehouse.orders)  # Expected output: {1: {'product_id': 1, 'quantity': 2,'status': 'done'}}

    # Test case for track_order
    print("\nTest case for track_order:")
    print(warehouse.track_order(1))  # Expected output: done
    print(warehouse.track_order(2))  # Expected output: False