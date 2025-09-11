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
        >>> complexCalculator = ComplexCalculator()
        >>> complexCalculator.add(1+2j, 3+4j)
        (4+6j)
        """
        return c1 + c2

    @staticmethod
    def subtract(c1, c2):
        """
        Subtracts two complex numbers.
        :param c1: The first complex number, complex.
        :param c2: The second complex number, complex.
        :return: The difference of the two complex numbers, complex.
        >>> complexCalculator = ComplexCalculator()
        >>> complexCalculator.subtract(1+2j, 3+4j)
        (-2-2j)
        """
        return c1 - c2

    @staticmethod
    def multiply(c1, c2):
        """
        Multiplies two complex numbers.
        :param c1: The first complex number, complex.
        :param c2: The second complex number, complex.
        :return: The product of the two complex numbers, complex.
        >>> complexCalculator = ComplexCalculator()
        >>> complexCalculator.multiply(1+2j, 3+4j)
        (-5+10j)
        """
        return c1 * c2

    @staticmethod
    def divide(c1, c2):
        """
        Divides two complex numbers.
        :param c1: The first complex number, complex.
        :param c2: The second complex number, complex.
        :return: The quotient of the two complex numbers, complex.
        >>> complexCalculator = ComplexCalculator()
        >>> complexCalculator.divide(1+2j, 3+4j)
        (0.44+0.08j)
        """
        if c2 == 0:
            raise ValueError("Cannot divide by zero.")
        return c1 / c2


if __name__ == "__main__":
    # Create an instance of ComplexCalculator
    complexCalculator = ComplexCalculator()
    
    # Test cases
    # Test addition
    output_add = complexCalculator.add(1+2j, 3+4j)
    print("Addition:", output_add)  # Expected output: (4+6j)
    
    # Test subtraction
    output_subtract = complexCalculator.subtract(1+2j, 3+4j)
    print("Subtraction:", output_subtract)  # Expected output: (-2-2j)
    
    # Test multiplication
    output_multiply = complexCalculator.multiply(1+2j, 3+4j)
    print("Multiplication:", output_multiply)  # Expected output: (-5+10j)
    
    # Test division
    output_divide = complexCalculator.divide(1+2j, 3+4j)
    print("Division:", output_divide)  # Expected output: (0.44+0.08j)