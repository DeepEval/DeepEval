class CurrencyConverter:
    """
    This is a class for currency conversion, which supports to convert amounts between different currencies, retrieve supported currencies, add new currency rates, and update existing currency rates.
    """

    def __init__(self):
        """
        Initialize the exchange rate of the US dollar against various currencies
        """
        self.rates = {
            'USD': 1.0,
            'EUR': 0.85,
            'GBP': 0.72,
            'JPY': 110.15,
            'CAD': 1.23,
            'AUD': 1.34,
            'CNY': 6.40,
        }

    def convert(self, amount, from_currency, to_currency):
        """
        Convert the value of a given currency to another currency type
        :param amount: float, The value of a given currency
        :param from_currency: string, source currency type
        :param to_currency: string, target currency type
        :return: float, value converted to another currency type
        """
        if from_currency not in self.rates or to_currency not in self.rates:
            raise ValueError("Currency not supported")

        # Convert from_currency to USD first
        amount_in_usd = amount / self.rates[from_currency]
        # Convert from USD to to_currency
        converted_amount = amount_in_usd * self.rates[to_currency]
        return converted_amount

    def get_supported_currencies(self):
        """
        Returns a list of supported currency types
        :return:list, All supported currency types
        """
        return list(self.rates.keys())

    def add_currency_rate(self, currency, rate):
        """
        Add a new supported currency type, return False if the currency type is already in the support list
        :param currency:string, currency type to be added
        :param rate:float, exchange rate for this type of currency
        :return:If successful, returns None; if unsuccessful, returns False
        """
        if currency in self.rates:
            return False
        self.rates[currency] = rate

    def update_currency_rate(self, currency, new_rate):
        """
        Update the exchange rate for a certain currency
        :param currency:string
        :param new_rate:float
        :return:If successful, returns None; if unsuccessful, returns False
        """
        if currency not in self.rates:
            return False
        self.rates[currency] = new_rate

# Test cases
if __name__ == "__main__":
    cc = CurrencyConverter()
    # Test convert method
    print(cc.convert(64, 'CNY', 'USD'))  # Expected output: 10.0

    # Test get_supported_currencies method
    print(cc.get_supported_currencies())  # Expected output: ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CNY']

    # Test add_currency_rate method
    print(cc.add_currency_rate('KRW', 1308.84))  # Expected output: None
    print(cc.add_currency_rate('EUR', 0.85))  # Expected output: False

    # Test update_currency_rate method
    print(cc.update_currency_rate('CNY', 7.18))  # Expected output: None
    print(cc.update_currency_rate('RUB', 0.015))  # Expected output: False