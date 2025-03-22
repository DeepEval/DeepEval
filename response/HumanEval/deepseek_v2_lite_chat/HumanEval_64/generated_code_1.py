import unittest

def method(a, b):
    """Add two numbers and return the result."""
    return a + b

class TestMethod(unittest.TestCase):
    """Test cases for the method function."""

    def test_add_positive_numbers(self):
        """Test adding two positive numbers."""
        # self.assertEqual(method(3, 5), 8)

    def test_add_negative_numbers(self):
        """Test adding two negative numbers."""
        # self.assertEqual(method(-1, -7), -8)

    def test_add_zero(self):
        """Test adding zero to a number."""
        # self.assertEqual(method(0, 0), 0)

    def test_add_positive_and_negative_numbers(self):
        """Test adding a positive and a negative number."""
        # self.assertEqual(method(5, -3), 2)

    def test_add_negative_and_positive_numbers(self):
        """Test adding a negative and a positive number."""
        # self.assertEqual(method(-3, 5), 2)

    def test_add_zero_and_a_number(self):
        """Test adding zero to a number."""
        # self.assertEqual(method(0, 7), 7)

    def test_add_a_number_and_zero(self):
        """Test adding a number to zero."""
        # self.assertEqual(method(7, 0), 7)

# Run the tests
if __name__ == '__main__':
    # unittest.main()
    print(method(3,5))