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
        >>> cc = CurrencyConverter()
        >>> cc.convert(64, 'CNY','USD')
        10.0
        """
        if from_currency not in self.rates or to_currency not in self.rates:
            raise ValueError("Unsupported currency type")
        # Convert to USD first, then to target currency
        usd_amount = amount / self.rates[from_currency]
        converted_amount = usd_amount * self.rates[to_currency]
        return converted_amount

    def get_supported_currencies(self):
        """
        Returns a list of supported currency types
        :return:list, All supported currency types
        >>> cc = CurrencyConverter()
        >>> cc.get_supported_currencies()
        ['USD','EUR','GBP','JPY','CAD','AUD','CNY']
        """
        return list(self.rates.keys())

    def add_currency_rate(self, currency, rate):
        """
        Add a new supported currency type, return False if the currency type is already in the support list
        :param currency:string, currency type to be added
        :param rate:float, exchange rate for this type of currency
        :return:If successful, returns None; if unsuccessful, returns False
        >>> cc = CurrencyConverter()
        >>> cc.add_currency_rate('KRW', 1308.84)
        >>> cc.rates['KRW']  # Check if the new rate has been added
        1308.84
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
        >>> cc = CurrencyConverter()
        >>> cc.update_currency_rate('CNY', 7.18)
        >>> cc.rates['CNY']  # Check if the rate has been updated
        7.18
        """
        if currency not in self.rates:
            return False
        self.rates[currency] = new_rate


if __name__ == "__main__":
    cc = CurrencyConverter()
    
    # Test convert method
    output = cc.convert(64, 'CNY', 'USD')
    print("Convert 64 CNY to USD:", output)  # Expecting 10.0

    # Test get_supported_currencies method
    output = cc.get_supported_currencies()
    print("Supported currencies:", output)  # Expecting ['USD','EUR','GBP','JPY','CAD','AUD','CNY']

    # Test add_currency_rate method
    cc.add_currency_rate('KRW', 1308.84)
    print("Added KRW rate:", cc.rates['KRW'])  # Expecting 1308.84

    # Test update_currency_rate method
    cc.update_currency_rate('CNY', 7.18)
    print("Updated CNY rate:", cc.rates['CNY'])  # Expecting 7.18