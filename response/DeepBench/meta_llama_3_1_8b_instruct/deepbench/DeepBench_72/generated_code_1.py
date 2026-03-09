import numpy as np
from math import gcd
from functools import reduce

def frequencies_to_period(frequencies, decimals=None):
    # Check if all frequencies are integers
    if not np.all(frequencies == frequencies.astype(int)):
        # If decimals is provided, round frequencies to the specified number of decimal places
        if decimals is not None:
            frequencies = np.round(frequencies, decimals)
        else:
            # If decimals is not provided, round frequencies to 6 decimal places by default
            frequencies = np.round(frequencies, 6)

    # Calculate the greatest common divisor of the frequencies
    freq_gcd = reduce(gcd, frequencies)

    # Calculate the period as 2π/freq_gcd
    period = 2 * np.pi / freq_gcd

    # Return the period as a tuple
    return (period,)

if __name__ == "__main__":
    # Create sample input values
    frequencies = np.array([1.2, 2.4, 3.6])
    decimals = 2

    # Call the function and print the results
    result = frequencies_to_period(frequencies, decimals)
    print("Period:", result[0])