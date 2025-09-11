import unittest

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
            raise ValueError("Unsupported currency")
        
        rate = self.rates[from_currency] / self.rates[to_currency]
        return amount * rate

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
        return None

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
        return None

# Test cases
class TestCurrencyConverter(unittest.TestCase):
    def setUp(self):
        self.cc = CurrencyConverter()

    def test_convert(self):
        self.assertAlmostEqual(self.cc.convert(100, 'USD', 'EUR'), 85.0)
        self.assertAlmostEqual(self.cc.convert(100, 'JPY', 'USD'), 0.89)

    def test_get_supported_currencies(self):
        self.assertEqual(self.cc.get_supported_currencies(), ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CNY'])

    def test_add_currency_rate(self):
        self.cc.add_currency_rate('KRW', 1000.0)
        self.assertEqual(self.cc.get_supported_currencies(), ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CNY', 'KRW'])

    def test_update_currency_rate(self):
        self.cc.update_currency_rate('CNY', 6.50)
        self.assertEqual(self.cc.rates['CNY'], 6.50)

if __name__ == "__main__":
    unittest.main()