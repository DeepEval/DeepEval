from math import gcd
from decimal import Decimal

def frequencies_to_period(frequencies, decimals=None):
    """Calculates the period of a Fourier series based on a set of frequencies.

    Args:
        frequencies: A tuple of frequencies.
        decimals: (Optional) The number of decimal places to round to.

    Returns:
        A tuple of frequencies.
    """

    # Convert frequencies to Decimal if necessary
    if not isinstance(frequencies, tuple):
        raise TypeError("Frequencies must be a tuple.")

    frequencies = tuple(Decimal(f) for f in frequencies)

    # Calculate the greatest common divisor (gcd) of the frequencies
    period = 2 * Decimal(3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679) / Decimal(gcd(*frequencies))

    # Round the period to the specified number of decimal places (if necessary)
    if decimals is not None:
        period = round(period, decimals)

    return period

if __name__ == "__main__":
    # Sample input values
    frequencies = (1, 2, 4)

    # Call the function and print the results
    period = frequencies_to_period(frequencies)
    print(period)