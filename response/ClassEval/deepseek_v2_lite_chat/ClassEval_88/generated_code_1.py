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
        :param x: float
        :return: float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.cos(60)
        0.5
        """
        x = x * pi / 180
        cos_x = 0.0
        sign = -1
        for i in range(n):
            cos_x += sign * (x**(2*i)) / fabs(factorial(2*i))
            sign *= -1
        return cos_x

    def factorial(self, a):
        """
        Calculate the factorial of a
        :param a: int
        :return: int
        >>> tricalculator.factorial(5)
        120
        """
        if a < 0:
            raise ValueError("Factorial is not defined for negative integers")
        if a == 0:
            return 1
        else:
            fact = 1
            for i in range(1, a + 1):
                fact = fact * i
            return fact

    def taylor(self, x, n):
        """
        Finding the n-order Taylor expansion value of cos (x/180 * pi)
        :param x: int
        :param n: int
        :return: float
        >>> tricalculator.taylor(60, 50)
        0.5000000000000001
        """
        x = x * pi / 180
        cos_x = 0.0
        sign = -1
        for i in range(n):
            cos_x += sign * (x**(2*i)) / fabs(self.factorial(2*i))
            sign *= -1
        return cos_x

    def sin(self, x):
        """
        Calculate the sin value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator.sin(30)
        0.5
        """
        # The sine of x is approximately equal to the value of the series
        # Sum(x^(2n + 1) / (2n + 1)!) for n from 0 to infinity.
        # This series converges to the sine function.
        x = x * pi / 180
        sin_x = 0.0
        sign = x
        for i in range(2*n + 1):
            sin_x += sign * (x**i) / self.factorial(2*i + 1)
            sign *= -1
        return sin_x

    def tan(self, x):
        """
        Calculate the tan value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator.tan(45)
        1.0
        """
        sin_x = self.sin(x)
        cos_x = self.cos(x)
        if cos_x == 0.0:
            raise ValueError("Division by zero error")
        return sin_x / cos_x

if __name__ == "__main__":
    tricalculator = TriCalculator()
    print(tricalculator.cos(60))  # 0.5
    print(tricalculator.sin(30))  # 0.5
    print(tricalculator.tan(45))  # 1.0