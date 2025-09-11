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
        return self.taylor(x, 50)  

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
            return a * self.factorial(a-1)

    def taylor(self, x, n):
        """
        Finding the n-order Taylor expansion value of cos (x/180 * pi)
        :param x: int
        :param n: int
        :return: float
        >>> tricalculator.taylor(60, 50)
        0.5000000000000001
        """
        result = 1
        for i in range(n+1):
            if i % 2 == 0:
                result += (x/180 * pi)**(2*i) / self.factorial(2*i)
            else:
                result -= (x/180 * pi)**(2*i+1) / self.factorial(2*i+1)
        return result

    def sin(self, x):
        """
        Calculate the sin value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator.sin(30)
        0.5
        """
        return self.taylor(x, 50)

    def tan(self, x):
        """
        Calculate the tan value of the x-degree angle
        :param x: float
        :return: float
        >>> tricalculator.tan(45)
        1.0
        """
        return self.sin(x) / self.cos(x)

if __name__ == "__main__":
    tricalculator = TriCalculator()
    # Test cases
    print(tricalculator.cos(60))  
    print(tricalculator.factorial(5))
    print(tricalculator.taylor(60, 50))
    print(tricalculator.sin(30))
    print(tricalculator.tan(45))