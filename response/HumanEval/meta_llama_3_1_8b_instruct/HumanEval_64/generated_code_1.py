import unittest

class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, num1, num2):
        """Return the sum of two numbers."""
        self.result = num1 + num2
        return self.result

    def subtract(self, num1, num2):
        """Return the difference of two numbers."""
        self.result = num1 - num2
        return self.result

    def multiply(self, num1, num2):
        """Return the product of two numbers."""
        self.result = num1 * num2
        return self.result

    def divide(self, num1, num2):
        """Return the quotient of two numbers."""
        if num2 == 0:
            raise ValueError("Cannot divide by zero.")
        self.result = num1 / num2
        return self.result


def method():
    calculator = Calculator()
    # Test cases
    calculator.add(5, 3)
    calculator.subtract(10, 4)
    calculator.multiply(7, 2)
    calculator.divide(9, 3)

    return calculator.result


class TestCalculator(unittest.TestCase):
    def test_add(self):
        calculator = Calculator()
        self.assertEqual(calculator.add(5, 3), 8)

    def test_subtract(self):
        calculator = Calculator()
        self.assertEqual(calculator.subtract(10, 4), 6)

    def test_multiply(self):
        calculator = Calculator()
        self.assertEqual(calculator.multiply(7, 2), 14)

    def test_divide(self):
        calculator = Calculator()
        self.assertEqual(calculator.divide(9, 3), 3)

    def test_divide_by_zero(self):
        calculator = Calculator()
        with self.assertRaises(ValueError):
            calculator.divide(5, 0)

    def test_invalid_input(self):
        calculator = Calculator()
        self.assertEqual(calculator.add('a', 3), 0)

if __name__ == '__main__':
    unittest.main()