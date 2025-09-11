import unittest

class CurrencyConverter:
    def __init__(self):
        self.rates = {'USD': 1.0, 'EUR': 0.85, 'GBP': 0.72, 'JPY': 110.15, 'CAD': 1.23, 'AUD': 1.34, 'CNY': 6.40}

    def convert(self, amount, from_currency, to_currency):
        if from_currency not in self.rates or to_currency not in self.rates:
            return None
        exchange_rate = self.rates[from_currency] / self.rates[to_currency]
        return amount * exchange_rate

    def get_supported_currencies(self):
        return list(self.rates.keys())

    def add_currency_rate(self, currency, rate):
        if currency in self.rates:
            return False
        self.rates[currency] = rate
        return None

    def update_currency_rate(self, currency, new_rate):
        if currency not in self.rates:
            return False
        self.rates[currency] = new_rate
        return None

class TestCurrencyConverter(unittest.TestCase):
    def test_convert(self):
        cc = CurrencyConverter()
        self.assertEqual(cc.convert(64, 'CNY', 'USD'), 10.0)

    def test_get_supported_currencies(self):
        cc = CurrencyConverter()
        self.assertEqual(cc.get_supported_currencies(), ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CNY'])

    def test_add_currency_rate(self):
        cc = CurrencyConverter()
        self.assertEqual(cc.add_currency_rate('KRW', 1308.84), None)
        self.assertEqual(cc.rates['KRW'], 1308.84)

    def test_update_currency_rate(self):
        cc = CurrencyConverter()
        self.assertEqual(cc.update_currency_rate('CNY', 7.18), None)
        self.assertEqual(cc.rates['CNY'], 7.18)

if __name__ == '__main__':
    unittest.main()