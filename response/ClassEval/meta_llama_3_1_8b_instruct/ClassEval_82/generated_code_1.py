class StockPortfolioTracker:
    """
    This is a class as StockPortfolioTracker that allows to add stocks, remove stocks, buy stocks, sell stocks, calculate the total value of the portfolio, and obtain a summary of the portfolio.
    """

    def __init__(self, cash_balance):
        """
        Initialize the StockPortfolioTracker class with a cash balance and an empty portfolio.
        """
        self.portfolio = []
        self.cash_balance = cash_balance

    def add_stock(self, stock):
        """
        Add a stock to the portfolio.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        """
        self.portfolio.append(stock)

    def remove_stock(self, stock):
        """
        Remove a stock from the portfolio.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        """
        self.portfolio = [s for s in self.portfolio if s!= stock]

    def buy_stock(self, stock, quantity):
        """
        Buy a stock and add it to the portfolio.
        :param stock: a dictionary with keys "name", "price"
        :param quantity: the quantity of the stock to buy,int.
        :return: True if the stock was bought successfully, False if the cash balance is not enough.
        """
        stock_price = stock["price"]
        total_cost = stock_price * quantity
        if self.cash_balance >= total_cost:
            self.cash_balance -= total_cost
            existing_stock = next((s for s in self.portfolio if s["name"] == stock["name"]), None)
            if existing_stock:
                existing_stock["quantity"] += quantity
            else:
                self.portfolio.append({"name": stock["name"], "price": stock_price, "quantity": quantity})
            return True
        return False

    def sell_stock(self, stock, quantity):
        """
        Sell a stock and remove it from the portfolio and add the cash to the cash balance.
        :param stock: a dictionary with keys "name", "price"
        :param quantity: the quantity of the stock to sell,int.
        :return: True if the stock was sold successfully, False if the quantity of the stock is not enough.
        """
        stock_price = stock["price"]
        total_revenue = stock_price * quantity
        existing_stock = next((s for s in self.portfolio if s["name"] == stock["name"]), None)
        if existing_stock and existing_stock["quantity"] >= quantity:
            self.cash_balance += total_revenue
            existing_stock["quantity"] -= quantity
            if existing_stock["quantity"] == 0:
                self.portfolio.remove(existing_stock)
            return True
        return False

    def calculate_portfolio_value(self):
        """
        Calculate the total value of the portfolio.
        :return: the total value of the portfolio, float.
        """
        return sum(s["price"] * s["quantity"] for s in self.portfolio) + self.cash_balance

    def get_portfolio_summary(self):
        """
        Get a summary of the portfolio.
        :return: a tuple of the total value of the portfolio and a list of dictionaries with keys "name" and "value"
        """
        portfolio_summary = [{"name": s["name"], "value": s["price"] * s["quantity"]} for s in self.portfolio]
        return (self.calculate_portfolio_value(), portfolio_summary)

    def get_stock_value(self, stock):
        """
        Get the value of a stock.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :return: the value of the stock, float.
        """
        existing_stock = next((s for s in self.portfolio if s["name"] == stock["name"]), None)
        if existing_stock:
            return existing_stock["price"] * existing_stock["quantity"]
        return 0

if __name__ == "__main__":
    # Test case
    tracker = StockPortfolioTracker(10000.0)

    # Test add_stock method
    tracker.add_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(tracker.portfolio)
    # Output: [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]

    # Test remove_stock method
    tracker.remove_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(tracker.portfolio)
    # Output: []

    # Test buy_stock method
    tracker.buy_stock({"name": "AAPL", "price": 150.0}, 10)
    print(tracker.portfolio)
    # Output: [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]
    print(tracker.cash_balance)
    # Output: 9500.0

    # Test sell_stock method
    tracker.sell_stock({"name": "AAPL", "price": 150.0}, 10)
    print(tracker.portfolio)
    # Output: []
    print(tracker.cash_balance)
    # Output: 15500.0

    # Test calculate_portfolio_value method
    print(tracker.calculate_portfolio_value())
    # Output: 15500.0

    # Test get_portfolio_summary method
    print(tracker.get_portfolio_summary())
    # Output: (15500.0, [{'name': 'AAPL', 'value': 1500.0}])

    # Test get_stock_value method
    print(tracker.get_stock_value({"name": "AAPL", "price": 150.0, "quantity": 10}))
    # Output: 1500.0