import math

class TriCalculator:
    """
    The class allows to calculate trigonometric values, including cosine, sine, and tangent, using Taylor series approximations.
    """

    def __init__(self):
        pass

    def cos(self, x):
        """
        Calculate the cos value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.cos(60)
        0.5
        """
        return math.cos(x * math.pi / 180)

    def factorial(self, a):
        """
        Calculate the factorial of a
        :param a: int
        :return: int
        >>> tricalculator.factorial(5)
        120
        """
        if a == 0:
            return 1
        else:
            return a * self.factorial(a - 1)

    def taylor(self, x, n):
        """
        Finding the n-order Taylor expansion value of cos (x/180 * pi)
        :param x: float
        :param n: int
        :return: float
        >>> tricalculator.taylor(60, 50)
        0.5000000000000001
        """
        return sum([(-1)**(i) * (x / 180 * math.pi)**(2*i) / self.factorial(2*i) for i in range(n)])

    def sin(self, x):
        """
        Calculate the sin value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator.sin(30)
        0.5
        """
        return math.sin(x * math.pi / 180)

    def tan(self, x):
        """
        Calculate the tan value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator.tan(45)
        1.0
        """
        return math.tan(x * math.pi / 180)
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    # Example usage
    tricalculator = TriCalculator()
    print("Cosine of 60 degrees:", tricalculator.cos(60))
    print("Sine of 30 degrees:", tricalculator.sin(30))
    print("Tangent of 45 degrees:", tricalculator.tan(45))
    print("Taylor expansion of cos(60) with 50 terms:", tricalculator.taylor(60, 50))
    print("Factorial of 5:", tricalculator.factorial(5))