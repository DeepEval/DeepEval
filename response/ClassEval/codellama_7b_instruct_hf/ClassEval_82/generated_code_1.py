import math

class StockPortfolioTracker:
    def __init__(self, cash_balance):
        self.portfolio = []
        self.cash_balance = cash_balance

    def add_stock(self, stock):
        if stock not in self.portfolio:
            self.portfolio.append(stock)
            return True
        else:
            return False

    def remove_stock(self, stock):
        if stock in self.portfolio:
            self.portfolio.remove(stock)
            return True
        else:
            return False

    def buy_stock(self, stock, quantity):
        if self.cash_balance >= stock['price'] * quantity:
            self.cash_balance -= stock['price'] * quantity
            stock['quantity'] += quantity
            return True
        else:
            return False

    def sell_stock(self, stock, quantity):
        if stock['quantity'] >= quantity:
            self.cash_balance += stock['price'] * quantity
            stock['quantity'] -= quantity
            return True
        else:
            return False

    def calculate_portfolio_value(self):
        total_value = 0
        for stock in self.portfolio:
            total_value += stock['price'] * stock['quantity']
        return total_value

    def get_portfolio_summary(self):
        total_value = self.calculate_portfolio_value()
        return total_value, [{'name': stock['name'], 'value': stock['price'] * stock['quantity']} for stock in self.portfolio]

    def get_stock_value(self, stock):
        return stock['price'] * stock['quantity']

if __name__ == "__main__":
    tracker = StockPortfolioTracker(10000.0)
    print(tracker.add_stock({"name": "AAPL", "price": 150.0, "quantity": 10}))
    print(tracker.portfolio)
    print(tracker.remove_stock({"name": "AAPL", "price": 150.0, "quantity": 10}))
    print(tracker.portfolio)
    print(tracker.buy_stock({"name": "AAPL", "price": 150.0, "quantity": 10}, 2))
    print(tracker.portfolio)
    print(tracker.sell_stock({"name": "AAPL", "price": 150.0, "quantity": 10}, 2))
    print(tracker.portfolio)
    print(tracker.calculate_portfolio_value())
    print(tracker.get_portfolio_summary())
    print(tracker.get_stock_value({"name": "AAPL", "price": 150.0, "quantity": 10}))