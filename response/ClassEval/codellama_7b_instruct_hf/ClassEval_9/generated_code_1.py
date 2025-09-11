import math

class BigNumCalculator:
    """
    This is a class that implements big number calculations, including adding, subtracting and multiplying.
    """

    @staticmethod
    def add(num1, num2):
        """
        Adds two big numbers.
        :param num1: The first number to add,str.
        :param num2: The second number to add,str.
        :return: The sum of the two numbers,str.
        """
        # Convert the numbers to integers
        num1 = int(num1)
        num2 = int(num2)

        # Add the numbers and convert the result back to a string
        result = str(num1 + num2)

        # Pad the result with leading zeros to match the length of the longest input number
        max_length = max(len(num1), len(num2))
        result = result.zfill(max_length)

        return result

    @staticmethod
    def subtract(num1, num2):
        """
        Subtracts two big numbers.
        :param num1: The first number to subtract,str.
        :param num2: The second number to subtract,str.
        :return: The difference of the two numbers,str.
        """
        # Convert the numbers to integers
        num1 = int(num1)
        num2 = int(num2)

        # Subtract the numbers and convert the result back to a string
        result = str(num1 - num2)

        # Pad the result with leading zeros to match the length of the longest input number
        max_length = max(len(num1), len(num2))
        result = result.zfill(max_length)

        return result

    @staticmethod
    def multiply(num1, num2):
        """
        Multiplies two big numbers.
        :param num1: The first number to multiply,str.
        :param num2: The second number to multiply,str.
        :return: The product of the two numbers,str.
        """
        # Convert the numbers to integers
        num1 = int(num1)
        num2 = int(num2)

        # Multiply the numbers and convert the result back to a string
        result = str(num1 * num2)

        # Pad the result with leading zeros to match the length of the longest input number
        max_length = max(len(num1), len(num2))
        result = result.zfill(max_length)

        return result

if __name__ == "__main__":
    bigNum = BigNumCalculator()
    print(bigNum.add("12345678901234567890", "98765432109876543210"))
    print(bigNum.subtract("12345678901234567890", "98765432109876543210"))
    print(bigNum.multiply("12345678901234567890", "98765432109876543210"))