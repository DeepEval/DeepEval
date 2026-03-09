import math
from functools import reduce
from fractions import Fraction

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def gcd_of_list(lst):
    return reduce(gcd, lst)

def frequencies_to_period(frequencies, decimals=0):
    if not frequencies:
        return ()

    rounded_frequencies = [round(f, decimals) for f in frequencies]
    integer_frequencies = [int(Fraction(f).limit_denominator(1000).denominator * f) for f in rounded_frequencies]
    common_gcd = gcd_of_list(integer_frequencies)
    
    period = 2 * math.pi / common_gcd
    
    return round(period, decimals)

if __name__ == "__main__":
    sample_frequencies = (1.0, 2.0, 3.0)
    period = frequencies_to_period(sample_frequencies, decimals=5)
    print("The calculated period is:", period)