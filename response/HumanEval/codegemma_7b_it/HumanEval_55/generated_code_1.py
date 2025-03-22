import unittest

def fibonacci(n):
    """
    Return nth Fibonacci number
    """
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

class FibonacciTest(unittest.TestCase):

    def test_fibonacci(self):
        self.assertEqual(fibonacci(0), 0)
        self.assertEqual(fibonacci(1), 1)
        self.assertEqual(fibonacci(6), 8)
        self.assertEqual(fibonacci(8), 21)


if __name__ == '__main__':
    unittest.main()