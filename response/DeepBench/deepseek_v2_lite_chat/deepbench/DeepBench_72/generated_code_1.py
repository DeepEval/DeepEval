from math import gcd
from fractions import Fraction

def frequencies_to_period(frequencies, decimals=2):
    # Round the frequencies to the specified number of decimal places
    rounded_frequencies = [round(Fraction(freq), decimals) for freq in frequencies]
    
    # Calculate the greatest common divisor of the frequencies
    gcd_value = gcd(*rounded_frequencies)
    
    # Calculate the period using the formula 2π/gcd(frequencies)
    periods = [Fraction(2 * pi) / Fraction(gcd_value) for pi in rounded_frequencies]
    
    return tuple(periods)

if __name__ == "__main__":
    frequencies = (1.5, 2.1, 3.7)
    decimals = 2
    periods = frequencies_to_period(frequencies, decimals)
    print("Periods:", periods)