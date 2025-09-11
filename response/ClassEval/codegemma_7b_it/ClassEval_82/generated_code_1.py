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
        >>> tracker = StockPortfolioTracker(10000.0)
        >>> tracker.add_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
        >>> tracker.portfolio
        [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]

        """
        self.portfolio.append(stock)

    def remove_stock(self, stock):
        """
        Remove a stock from the portfolio.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        >>> tracker = StockPortfolioTracker(10000.0)
        >>> tracker.portfolio = [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]
        >>> tracker.remove_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
        True
        >>> tracker.portfolio
        []

        """
        self.portfolio.remove(stock)

    def buy_stock(self, stock):
        """
        Buy a stock and add it to the portfolio.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :param quantity: the quantity of the stock to buy,int.
        :return: True if the stock was bought successfully, False if the cash balance is not enough.
        >>> tracker = StockPortfolioTracker(10000.0)
        >>> tracker.buy_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
        True
        >>> tracker.portfolio
        [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]

        """
        if self.cash_balance < stock["price"] * stock["quantity"]:
            return False
        self.cash_balance -= stock["price"] * stock["quantity"]
        self.add_stock(stock)
        return True

    def sell_stock(self, stock):
        """
        Sell a stock and remove it from the portfolio and add the cash to the cash balance.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :param quantity: the quantity of the stock to sell,int.
        :return: True if the stock was sold successfully, False if the quantity of the stock is not enough.
        >>> tracker = StockPortfolioTracker(10000.0)
        >>> tracker.portfolio = [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]
        >>> tracker.sell_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
        True
        >>> tracker.portfolio
        []

        """
        if stock not in self.portfolio or stock["quantity"] > self.portfolio[self.portfolio.index(stock)]["quantity"]:
            return False
        self.cash_balance += stock["price"] * stock["quantity"]
        self.remove_stock(stock)
        return True

    def calculate_portfolio_value(self):
        """
        Calculate the total value of the portfolio.
        :return: the total value of the portfolio, float.
        >>> tracker = StockPortfolioTracker(10000.0)
        >>> tracker.portfolio = [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]
        >>> tracker.calculate_portfolio_value()
        11500.0

        """
        total_value = 0
        for stock in self.portfolio:
            total_value += stock["price"] * stock["quantity"]
        return total_value + self.cash_balance

    def get_portfolio_summary(self):
        """
        Get a summary of the portfolio.
        :return: a tuple of the total value of the portfolio and a list of dictionaries with keys "name" and "value"
        >>> tracker = StockPortfolioTracker(10000.0)
        >>> tracker.portfolio = [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]
        >>> tracker.get_portfolio_summary()
        (11500.0, [{'name': 'AAPL', 'value': 1500.0}])

        """
        total_value = self.calculate_portfolio_value()
        portfolio_summary = []
        for stock in self.portfolio:
            portfolio_summary.append({"name": stock["name"], "value": stock["price"] * stock["quantity"]})
        return total_value, portfolio_summary

    def get_stock_value(self, stock):
        """
        Get the value of a stock.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :return: the value of the stock, float.
        >>> tracker = StockPortfolioTracker(10000.0)
        >>> tracker.get_stock_value({"name": "AAPL", "price": 150.0, "quantity": 10})
        1500.0

        """
        if stock in self.portfolio:
            return stock["price"] * stock["quantity"]
        return 0

if __name__ == "__main__":
    tracker = StockPortfolioTracker(10000.0)
    tracker.add_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(tracker.portfolio)
    tracker.remove_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(tracker.portfolio)
    tracker.buy_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(tracker.portfolio)
    tracker.sell_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(tracker.portfolio)
    print(tracker.cash_balance)
    print(tracker.calculate_portfolio_value())
    print(tracker.get_portfolio_summary())
    print(tracker.get_stock_value({"name": "AAPL", "price": 150.0, "quantity": 10}))