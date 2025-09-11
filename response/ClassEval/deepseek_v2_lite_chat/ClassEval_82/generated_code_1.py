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
        return True

    def remove_stock(self, stock):
        """
        Remove a stock from the portfolio.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        """
        if self.check_stock_in_portfolio(stock):
            self.portfolio.remove(stock)
            return True
        else:
            return False

    def check_stock_in_portfolio(self, stock):
        """
        Check if a stock is in the portfolio.
        """
        for stock_item in self.portfolio:
            if stock['name'] == stock_item['name']:
                return True
        return False

    def buy_stock(self, stock):
        """
        Buy a stock and add it to the portfolio.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :param quantity: the quantity of the stock to buy,int.
        :return: True if the stock was bought successfully, False if the cash balance is not enough.
        """
        required_cash = stock['price'] * stock['quantity']
        if self.cash_balance >= required_cash:
            self.portfolio.append(stock)
            self.cash_balance -= required_cash
            return True
        else:
            return False

    def sell_stock(self, stock):
        """
        Sell a stock and remove it from the portfolio and add the cash to the cash balance.
        :param stock: a dictionary with keys "name", "price", and "quantity"
        :param quantity: the quantity of the stock to sell,int.
        :return: True if the stock was sold successfully, False if the quantity of the stock is not enough.
        """
        for stock_item in self.portfolio:
            if stock['name'] == stock_item['name'] and stock['quantity'] <= stock_item['quantity']:
                cash_earned = stock['price'] * stock['quantity']
                self.cash_balance += cash_earned
                self.portfolio.remove(stock)
                return True
        return False

    def calculate_portfolio_value(self):
        """
        Calculate the total value of the portfolio.
        :return: the total value of the portfolio, float.
        """
        total_value = 0.0
        for stock in self.portfolio:
            total_value += stock['price'] * stock['quantity']
        return total_value

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
        for item in self.portfolio:
            if stock['name'] == item['name']:
                return item['price'] * stock['quantity']
        return 0.0

# Test cases
if __name__ == "__main__":
    tracker = StockPortfolioTracker(10000.0)
    tracker.add_stock({"name": "AAPL", "price": 150.0, "quantity": 10})
    print(tracker.remove_stock({"name": "AAPL", "price": 150.0, "quantity": 10}))  # True
    print(tracker.check_stock_in_portfolio({"name": "AAPL", "price": 150.0, "quantity": 10}))  # False
    print(tracker.buy_stock({"name": "AAPL", "price": 150.0, "quantity": 10}))  # True
    print(tracker.calculate_portfolio_value())  # 15000.0
    print(tracker.sell_stock({"name": "AAPL", "price": 150.0, "quantity": 10}))  # True
    print(tracker.get_portfolio_summary())  # (15000.0, [{'name': 'AAPL', 'value': 1500.0}])
    print(tracker.get_stock_value({"name": "AAPL", "price": 150.0, "quantity": 10}))  # 1500.0