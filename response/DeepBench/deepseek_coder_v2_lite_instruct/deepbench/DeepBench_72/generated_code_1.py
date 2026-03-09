import math
from fractions import Fraction

def frequencies_to_period(frequencies, decimals=None):
    if decimals is not None:
        frequencies = tuple(round(freq, decimals) for freq in frequencies)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    common_divisor = math.gcd(*frequencies)
    period = 2 * math.pi / common_divisor
    return period

if __name__ == "__main__":
    frequencies = (1, 2, 3)
    decimals = 2
    period = frequencies_to_period(frequencies, decimals)
    print(f"The period of the Fourier series is: {period}")