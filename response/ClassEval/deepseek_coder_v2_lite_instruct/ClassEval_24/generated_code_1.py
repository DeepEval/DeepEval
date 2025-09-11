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
        return c1 / c2

if __name__ == "__main__":
    complexCalculator = ComplexCalculator()
    
    # Test cases
    print("Addition:")
    print(complexCalculator.add(1+2j, 3+4j))  # Expected output: (4+6j)
    
    print("Subtraction:")
    print(complexCalculator.subtract(1+2j, 3+4j))  # Expected output: (-2-2j)
    
    print("Multiplication:")
    print(complexCalculator.multiply(1+2j, 3+4j))  # Expected output: (-5+10j)
    
    print("Division:")
    print(complexCalculator.divide(1+2j, 3+4j))  # Expected output: (0.44+0.08j)