from math import pi, fabs

class TriCalculator:
    """
    The class allows to calculate trigonometric values, including cosine, sine, and tangent, using Taylor series approximations.
    """

    def __init__(self):
        pass

    def factorial(self, a):
        """
        Calculate the factorial of a
        :param a: int
        :return: int
        >>> tricalculator = TriCalculator()
        >>> tricalculator.factorial(5)
        120
        """
        if a == 0 or a == 1:
            return 1
        return a * self.factorial(a - 1)

    def cos(self, x):
        """
        Calculate the cos value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.cos(60)
        0.5
        """
        return self.taylor(x, 10)

    def sin(self, x):
        """
        Calculate the sin value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.sin(30)
        0.5
        """
        return self.taylor(x, 10, is_sine=True)

    def taylor(self, x, n, is_sine=False):
        """
        Finding the n-order Taylor expansion value of cos (x/180 * pi)
        :param x: int
        :param n: int
        :param is_sine: bool - flag to determine if calculating sine
        :return: float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.taylor(60, 50)
        0.5000000000000001
        """
        rad = x * (pi / 180)  # Convert degrees to radians
        result = 0.0
        
        for i in range(n):
            term = (rad**(2*i)) / self.factorial(2*i) if not is_sine else (rad**(2*i + 1)) / self.factorial(2*i + 1)
            result += term if not is_sine else (-1)**i * term

        return result

    def tan(self, x):
        """
        Calculate the tan value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator = TriCalculator()
        >>> tricalculator.tan(45)
        1.0
        """
        sin_value = self.sin(x)
        cos_value = self.cos(x)
        if fabs(cos_value) < 1e-10:
            raise ValueError("Tangent is undefined for angles where cosine is zero.")
        return sin_value / cos_value

# Testing the implementation
if __name__ == "__main__":
    tricalculator = TriCalculator()
    
    # Test case for factorial
    output_factorial = tricalculator.factorial(5)
    print(f"Factorial of 5: {output_factorial}")  # Expected: 120

    # Test case for cosine
    output_cos = tricalculator.cos(60)
    print(f"Cosine of 60 degrees: {output_cos}")  # Expected: 0.5

    # Test case for sine
    output_sin = tricalculator.sin(30)
    print(f"Sine of 30 degrees: {output_sin}")  # Expected: 0.5

    # Test case for tangent
    output_tan = tricalculator.tan(45)
    print(f"Tangent of 45 degrees: {output_tan}")  # Expected: 1.0