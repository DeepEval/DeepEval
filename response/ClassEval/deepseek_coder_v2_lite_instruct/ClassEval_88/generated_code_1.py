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
        """
        return self._taylor_cos(x / 180 * pi, 10)

    def factorial(self, a):
        """
        Calculate the factorial of a
        :param a: int
        :return: int
        """
        if a == 0:
            return 1
        result = 1
        for i in range(1, a + 1):
            result *= i
        return result

    def _taylor_cos(self, x, n):
        """
        Finding the n-order Taylor expansion value of cos(x)
        :param x: float
        :param n: int
        :return: float
        """
        result = 0
        for i in range(n):
            coef = (-1) ** i
            num = x ** (2 * i)
            denom = self.factorial(2 * i)
            result += coef * num / denom
        return result

    def sin(self, x):
        """
        Calculate the sin value of the x-degree angle
        :param x: float
        :return: float
        """
        return self._taylor_sin(x / 180 * pi, 10)

    def _taylor_sin(self, x, n):
        """
        Finding the n-order Taylor expansion value of sin(x)
        :param x: float
        :param n: int
        :return: float
        """
        result = 0
        for i in range(n):
            coef = (-1) ** i
            num = x ** (2 * i + 1)
            denom = self.factorial(2 * i + 1)
            result += coef * num / denom
        return result

    def tan(self, x):
        """
        Calculate the tan value of the x-degree angle
        :param x: float
        :return: float
        """
        cos_val = self.cos(x)
        if fabs(cos_val) < 1e-10:
            raise ValueError("Cannot compute tangent for cos(x) = 0")
        return self.sin(x) / cos_val

if __name__ == "__main__":
    tricalculator = TriCalculator()
    
    # Test cases
    print("cos(60):", tricalculator.cos(60))  # Expected output: 0.5
    print("sin(30):", tricalculator.sin(30))  # Expected output: 0.5
    print("tan(45):", tricalculator.tan(45))  # Expected output: 1.0
    print("factorial(5):", tricalculator.factorial(5))  # Expected output: 120
    print("taylor(60, 50):", tricalculator.taylor(60, 50))  # Expected output: 0.5000000000000001