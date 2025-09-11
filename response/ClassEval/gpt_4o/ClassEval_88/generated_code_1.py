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
        return self.taylor(x, 50)

    def factorial(self, a):
        """
        Calculate the factorial of a
        :param a: int
        :return: int
        """
        if a == 0:
            return 1
        result = 1
        for i in range(2, a + 1):
            result *= i
        return result

    def taylor(self, x, n):
        """
        Finding the n-order Taylor expansion value of cos (x/180 * pi)
        :param x: int
        :param n: int
        :return: float
        """
        # Convert degrees to radians
        x_rad = x * pi / 180
        result = 0
        for i in range(n):
            result += ((-1)**i) * (x_rad**(2*i)) / self.factorial(2*i)
        return result

    def sin(self, x):
        """
        Calculate the sin value of the x-degree angle
        :param x: float
        :return: float
        """
        # Use the sine Taylor series expansion
        x_rad = x * pi / 180
        result = 0
        for i in range(50):  # using 50 terms for good approximation
            result += ((-1)**i) * (x_rad**(2*i + 1)) / self.factorial(2*i + 1)
        return result

    def tan(self, x):
        """
        Calculate the tan value of the x-degree angle
        :param x: float
        :return: float
        """
        cos_value = self.cos(x)
        sin_value = self.sin(x)
        if fabs(cos_value) < 1e-15:  # Handle the case where cos is very close to 0
            raise ValueError("Tangent undefined for this angle.")
        return sin_value / cos_value

# Test cases
if __name__ == "__main__":
    tricalculator = TriCalculator()

    # Test case for cos
    output_cos = tricalculator.cos(60)
    print(f"cos(60): {output_cos}")  # Expected: ~0.5

    # Test case for factorial
    output_factorial = tricalculator.factorial(5)
    print(f"factorial(5): {output_factorial}")  # Expected: 120

    # Test case for taylor (cos(60))
    output_taylor = tricalculator.taylor(60, 50)
    print(f"taylor(60, 50): {output_taylor}")  # Expected: ~0.5

    # Test case for sin
    output_sin = tricalculator.sin(30)
    print(f"sin(30): {output_sin}")  # Expected: ~0.5

    # Test case for tan
    output_tan = tricalculator.tan(45)
    print(f"tan(45): {output_tan}")  # Expected: ~1.0