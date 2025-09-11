class BigNumCalculator:
    """
    This is a class that implements big number calculations, including adding, subtracting and multiplying.
    """

    @staticmethod
    def add(num1, num2):
        """
        Adds two big numbers.
        :param num1: The first number to add, str.
        :param num2: The second number to add, str.
        :return: The sum of the two numbers, str.
        """
        return str(int(num1) + int(num2))

    @staticmethod
    def subtract(num1, num2):
        """
        Subtracts two big numbers.
        :param num1: The first number to subtract, str.
        :param num2: The second number to subtract, str.
        :return: The difference of the two numbers, str.
        """
        return str(int(num1) - int(num2))

    @staticmethod
    def multiply(num1, num2):
        """
        Multiplies two big numbers.
        :param num1: The first number to multiply, str.
        :param num2: The second number to multiply, str.
        :return: The product of the two numbers, str.
        """
        return str(int(num1) * int(num2))


if __name__ == "__main__":
    bigNum = BigNumCalculator()

    # Test case for add method
    output_add = bigNum.add("12345678901234567890", "98765432109876543210")
    print("Addition result:", output_add)  # Expected: '111111111011111111100'

    # Test case for subtract method
    output_subtract = bigNum.subtract("12345678901234567890", "98765432109876543210")
    print("Subtraction result:", output_subtract)  # Expected: '-86419753208641975320'

    # Test case for multiply method
    output_multiply = bigNum.multiply("12345678901234567890", "98765432109876543210")
    print("Multiplication result:", output_multiply)  # Expected: '1219326311370217952237463801111263526900'