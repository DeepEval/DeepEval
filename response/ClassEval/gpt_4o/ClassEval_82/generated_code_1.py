class StockPortfolioTracker:
    """
    This is a class as StockPortfolioTracker that allows to add stocks, remove stocks, 
    buy stocks, sell stocks, calculate the total value of the portfolio, and obtain a summary of the portfolio.
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
        try:
            self.portfolio.remove(stock)
            return True
        except ValueError:
            return False

    def buy_stock(self, stock):
        """
        Buy a stock and add it to the portfolio.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :return: True if the stock was bought successfully, False if the cash balance is not enough.
        """
        total_price = stock['price'] * stock['quantity']
        if self.cash_balance >= total_price:
            self.cash_balance -= total_price
            self.add_stock(stock)
            return True
        else:
            return False

    def sell_stock(self, stock):
        """
        Sell a stock and remove it from the portfolio and add the cash to the cash balance.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :return: True if the stock was sold successfully, False if the quantity of the stock is not enough.
        """
        for existing_stock in self.portfolio:
            if existing_stock['name'] == stock['name'] and existing_stock['quantity'] >= stock['quantity']:
                existing_stock['quantity'] -= stock['quantity']
                self.cash_balance += stock['price'] * stock['quantity']
                if existing_stock['quantity'] == 0:
                    self.portfolio.remove(existing_stock)
                return True
        return False

    def calculate_portfolio_value(self):
        """
        Calculate the total value of the portfolio.
        :return: the total value of the portfolio, float.
        """
        portfolio_value = sum(stock['price'] * stock['quantity'] for stock in self.portfolio)
        return self.cash_balance + portfolio_value

    def get_portfolio_summary(self):
        """
        Get a summary of the portfolio.
        :return: a tuple of the total value of the portfolio and a list of dictionaries with keys "name" and "value"
        """
        total_value = self.calculate_portfolio_value()
        summary = [{'name': stock['name'], 'value': stock['price'] * stock['quantity']} for stock in self.portfolio]
        return total_value, summary

    def get_stock_value(self, stock):
        """
        Get the value of a stock.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :return: the value of the stock, float.
        """
        return stock['price'] * stock['quantity']


if __name__ == "__main__":
    # Test case for add_stock
    tracker = StockPortfolioTracker(10000.0)
    tracker.add_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(tracker.portfolio)  # [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]
    
    # Test case for remove_stock
    result = tracker.remove_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(result)  # True
    print(tracker.portfolio)  # []
    
    # Test case for buy_stock
    tracker = StockPortfolioTracker(10000.0)
    result = tracker.buy_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(result)  # True
    print(tracker.portfolio)  # [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]
    
    # Test case for sell_stock
    result = tracker.sell_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(result)  # True
    print(tracker.portfolio)  # []
    
    # Test case for calculate_portfolio_value
    tracker = StockPortfolioTracker(10000.0)
    tracker.portfolio = [{'name': 'AAPL', 'price': 150.0, 'quantity': 10}]
    value = tracker.calculate_portfolio_value()
    print(value)  # 11500.0
    
    # Test case for get_portfolio_summary
    summary = tracker.get_portfolio_summary()
    print(summary)  # (11500.0, [{'name': 'AAPL', 'value': 1500.0}])
    
    # Test case for get_stock_value
    value = tracker.get_stock_value({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(value)  # 1500.0