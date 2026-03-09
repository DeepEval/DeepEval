from numpy import sum as np_sum
import numpy as np
def hamming_distance(hash1, hash2):
    hash1 = hash1.ljust(64, '0')
    hash2 = hash2.ljust(64, '0')
    return float(np_sum(np_sum((np.array(list(hash1)) != np.array(list(hash2))) == 1, axis=0))) / 64

if __name__ == "__main__":
    hash1 = "a0f523f3290c1b624732172a04f0f390"
    hash2 = "a0f523f3290c1b624732172a04f0f391"
    print(hamming_distance(hash1, hash2))