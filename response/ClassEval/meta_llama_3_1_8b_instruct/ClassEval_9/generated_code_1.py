import math

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
        >>> bigNum = BigNumCalculator()
        >>> bigNum.add("12345678901234567890", "98765432109876543210")
        '111111111011111111100'
        """
        # Convert the input strings to integers, then convert the result back to a string
        result = str(int(num1) + int(num2))
        return result

    @staticmethod
    def subtract(num1, num2):
        """
        Subtracts two big numbers.
        :param num1: The first number to subtract, str.
        :param num2: The second number to subtract, str.
        :return: The difference of the two numbers, str.
        >>> bigNum = BigNumCalculator()
        >>> bigNum.subtract("12345678901234567890", "98765432109876543210")
        '-86419753208641975320'
        """
        # Convert the input strings to integers, then convert the result back to a string
        result = str(int(num1) - int(num2))
        # If the result is negative, add a negative sign to the result
        if result[0] == '-':
            return result
        else:
            return '-' + result

    @staticmethod
    def multiply(num1, num2):
        """
        Multiplies two big numbers.
        :param num1: The first number to multiply, str.
        :param num2: The second number to multiply, str.
        :return: The product of the two numbers, str.
        >>> bigNum = BigNumCalculator()
        >>> bigNum.multiply("12345678901234567890", "98765432109876543210")
        '1219326311370217952237463801111263526900'
        """
        # Convert the input strings to integers, then convert the result back to a string
        result = str(int(num1) * int(num2))
        return result

if __name__ == "__main__":
    big_num = BigNumCalculator()
    print(big_num.add("12345678901234567890", "98765432109876543210"))
    print(big_num.subtract("12345678901234567890", "98765432109876543210"))
    print(big_num.multiply("12345678901234567890", "98765432109876543210"))