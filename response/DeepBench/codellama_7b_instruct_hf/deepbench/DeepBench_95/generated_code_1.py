import numpy as np

def hamming_distance(hash1, hash2):
    # Pad the length of the hashes to be 64 bits
    hash1 = hash1.zfill(64)
    hash2 = hash2.zfill(64)

    # Calculate the Hamming distance
    hamming_distance = np.count_nonzero(hash1 != hash2)

    return hamming_distance

if __name__ == "__main__":
    # Create sample input values
    hash1 = "10010101010101010101010101010101"
    hash2 = "11010101010101010101010101010101"

    # Call the function and print the result
    print(hamming_distance(hash1, hash2))