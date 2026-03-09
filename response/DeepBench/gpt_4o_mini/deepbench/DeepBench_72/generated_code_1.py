import numpy as np
from math import gcd
from functools import reduce

def frequencies_to_period(frequencies, decimals=2): 
    rounded_frequencies = [round(f, decimals) for f in frequencies]
    common_divisor = reduce(gcd, rounded_frequencies)
    period = 2 * np.pi / common_divisor if common_divisor != 0 else float('inf')
    return tuple(rounded_frequencies), period

if __name__ == "__main__":
    sample_frequencies = (3.14, 6.28, 9.42)
    result = frequencies_to_period(sample_frequencies, decimals=2)
    print("Frequencies:", result[0])
    print("Period:", result[1])