class ComplexCalculator:
    """
    This is a class that implements addition, subtraction, multiplication, and division operations for complex numbers.
    """

    def __init__(self):
        pass

    @staticmethod
    def add(c1, c2):
        """
        Adds two complex numbers.
        :param c1: The first complex number, complex.
        :param c2: The second complex number, complex.
        :return: The sum of the two complex numbers, complex.
        """
        return c1 + c2

    @staticmethod
    def subtract(c1, c2):
        """
        Subtracts two complex numbers.
        :param c1: The first complex number, complex.
        :param c2: The second complex number, complex.
        :return: The difference of the two complex numbers, complex.
        """
        return c1 - c2

    @staticmethod
    def multiply(c1, c2):
        """
        Multiplies two complex numbers.
        :param c1: The first complex number, complex.
        :param c2: The second complex number, complex.
        :return: The product of the two complex numbers, complex.
        """
        return c1 * c2

    @staticmethod
    def divide(c1, c2):
        """
        Divides two complex numbers.
        :param c1: The first complex number, complex.
        :param c2: The second complex number, complex.
        :return: The quotient of the two complex numbers, complex.
        """
        if c2 == 0:
            raise ValueError("Cannot divide by zero")
        return c1 / c2

if __name__ == "__main__":
    complexCalculator = ComplexCalculator()

    # Test case for addition
    result_add = complexCalculator.add(1+2j, 3+4j)
    print(f"Addition Result: {result_add}")  # Output: (4+6j)

    # Test case for subtraction
    result_subtract = complexCalculator.subtract(1+2j, 3+4j)
    print(f"Subtraction Result: {result_subtract}")  # Output: (-2-2j)

    # Test case for multiplication
    result_multiply = complexCalculator.multiply(1+2j, 3+4j)
    print(f"Multiplication Result: {result_multiply}")  # Output: (-5+10j)

    # Test case for division
    result_divide = complexCalculator.divide(1+2j, 3+4j)
    print(f"Division Result: {result_divide}")  # Output: (0.44+0.08j)