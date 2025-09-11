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
        :return: True if the stock was removed successfully, False otherwise
        """
        try:
            self.portfolio.remove(stock)
            return True
        except ValueError:
            return False

    def buy_stock(self, stock):
        """
        Buy a stock and add it to the portfolio.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :return: True if the stock was bought successfully, False if the cash balance is not enough
        """
        total_cost = stock['price'] * stock['quantity']
        if self.cash_balance >= total_cost:
            self.cash_balance -= total_cost
            self.add_stock(stock)
            return True
        return False

    def sell_stock(self, stock):
        """
        Sell a stock and remove it from the portfolio and add the cash to the cash balance.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :return: True if the stock was sold successfully, False if the quantity of the stock is not enough
        """
        if stock['quantity'] <= self.get_stock_quantity(stock):
            self.cash_balance += stock['price'] * stock['quantity']
            self.remove_stock(stock)
            return True
        return False

    def calculate_portfolio_value(self):
        """
        Calculate the total value of the portfolio.
        :return: the total value of the portfolio, float
        """
        total_value = self.cash_balance
        for stock in self.portfolio:
            total_value += stock['price'] * stock['quantity']
        return total_value

    def get_portfolio_summary(self):
        """
        Get a summary of the portfolio.
        :return: a tuple of the total value of the portfolio and a list of dictionaries with keys "name" and "value"
        """
        summary = []
        for stock in self.portfolio:
            value = stock['price'] * stock['quantity']
            summary.append({'name': stock['name'], 'value': value})
        return (self.calculate_portfolio_value(), summary)

    def get_stock_value(self, stock):
        """
        Get the value of a stock.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :return: the value of the stock, float
        """
        return stock['price'] * stock['quantity']

    def get_stock_quantity(self, stock):
        """
        Get the quantity of a stock in the portfolio.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :return: the quantity of the stock, int
        """
        for item in self.portfolio:
            if item['name'] == stock['name']:
                return item['quantity']
        return 0

if __name__ == "__main__":
    # Test cases
    tracker = StockPortfolioTracker(10000.0)

    # Test add_stock
    tracker.add_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(tracker.portfolio)  # [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]

    # Test remove_stock
    removed = tracker.remove_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(removed)  # True
    print(tracker.portfolio)  # []

    # Test buy_stock
    bought = tracker.buy_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(bought)  # True
    print(tracker.portfolio)  # [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]

    # Test sell_stock
    sold = tracker.sell_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(sold)  # True
    print(tracker.portfolio)  # []

    # Test calculate_portfolio_value
    tracker.portfolio = [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]
    value = tracker.calculate_portfolio_value()
    print(value)  # 11500.0

    # Test get_portfolio_summary
    summary = tracker.get_portfolio_summary()
    print(summary)  # (11500.0, [{'name': 'AAPL', 'value': 1500.0}])

    # Test get_stock_value
    stock_value = tracker.get_stock_value({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(stock_value)  # 1500.0