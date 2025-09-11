import cmath

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
            raise ZeroDivisionError("Cannot divide by zero")
        return c1 / c2


if __name__ == "__main__":
    complex_calculator = ComplexCalculator()

    # Test case for add method
    c1 = 1 + 2j
    c2 = 3 + 4j
    output = complex_calculator.add(c1, c2)
    print("Addition: ", output)

    # Test case for subtract method
    c1 = 1 + 2j
    c2 = 3 + 4j
    output = complex_calculator.subtract(c1, c2)
    print("Subtraction: ", output)

    # Test case for multiply method
    c1 = 1 + 2j
    c2 = 3 + 4j
    output = complex_calculator.multiply(c1, c2)
    print("Multiplication: ", output)

    # Test case for divide method
    c1 = 1 + 2j
    c2 = 3 + 4j
    output = complex_calculator.divide(c1, c2)
    print("Division: ", output)

    # Test case for divide by zero
    try:
        c1 = 1 + 2j
        c2 = 0
        output = complex_calculator.divide(c1, c2)
        print(output)
    except ZeroDivisionError as e:
        print(e)