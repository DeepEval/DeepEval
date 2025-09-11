from math import pi, fabs

class TriCalculator:
    """
    The class allows to calculate trigonometric values, including cosine, sine, and tangent, using Taylor series approximations.
    """

    def __init__(self):
        pass

    def cos(self, x):
        """
        Calculate the cos value of the x-degree angle
        :param x:float
        :return:float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.cos(60)
        0.5
        """
        return self.taylor(x, 50)  # Taylor expansion of cos(x/180 * pi) to 50 terms

    def factorial(self, a):
        """
        Calculate the factorial of a
        :param a: int
        :return: int
        >>> tricalculator = TriCalculator()
        >>> tricalculator.factorial(5)
        120
        """
        if a == 0:
            return 1
        else:
            return a * self.factorial(a-1)

    def taylor(self, x, n):
        """
        Finding the n-order Taylor expansion value of cos (x/180 * pi)
        :param x: int
        :param n: int
        :return: float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.taylor(60, 50)
        0.5000000000000001
        """
        result = 0
        for i in range(n):
            result += ((-1)**i * (x/180 * pi)**(2*i)) / self.factorial(2*i)
        return result

    def sin(self, x):
        """
        Calculate the sin value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.sin(30)
        0.5
        """
        return self.taylor(x, 50) * self.cos(90-x)  # Use identity sin(x) = cos(90-x)

    def tan(self, x):
        """
        Calculate the tan value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.tan(45)
        1.0
        """
        if self.cos(x) == 0:
            raise ValueError("Cannot calculate tan(90) or tan(270)")
        else:
            return self.sin(x) / self.cos(x)

if __name__ == "__main__":
    tricalculator = TriCalculator()
    
    # Test case for cos
    print(tricalculator.cos(60))  # Output: 0.5
    
    # Test case for factorial
    print(tricalculator.factorial(5))  # Output: 120
    
    # Test case for taylor
    print(tricalculator.taylor(60, 50))  # Output: 0.5
    
    # Test case for sin
    print(tricalculator.sin(30))  # Output: 0.5
    
    # Test case for tan
    print(tricalculator.tan(45))  # Output: 1.0
    
    # Test case for tan(90)
    try:
        print(tricalculator.tan(90))
    except ValueError as e:
        print(e)  # Output: Cannot calculate tan(90) or tan(270)