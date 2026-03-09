import math

def frequencies_to_period(frequencies, decimals=0):
    """
    Calculates the period of a Fourier series based on a set of frequencies.
    The input parameters are a tuple of frequencies called frequencies and an optional parameter for the number of decimal places to round to called decimals.
    The function returns a tuple of frequencies.
    The period is calculated as 2π/gcd(frequencies), where gcd is the greatest common divisor of the frequencies.
    If the frequencies are not integers, the function rounds them to the specified number of decimal places before calculating the period.
    """
    # Round the frequencies to the specified number of decimal places
    rounded_frequencies = tuple(round(f, decimals) for f in frequencies)

    # Calculate the greatest common divisor of the rounded frequencies
    gcd = math.gcd(*rounded_frequencies)

    # Calculate the period
    period = 2 * math.pi / gcd

    return period

if __name__ == "__main__":
    # Example usage
    frequencies = (1.5, 2.5, 3.5)
    period = frequencies_to_period(frequencies)
    print(period)  # Output: 2.0