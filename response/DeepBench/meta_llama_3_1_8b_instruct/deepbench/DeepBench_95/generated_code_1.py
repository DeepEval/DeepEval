import math

def hamming_distance(hash1, hash2):
    # Pad the length to be 64 for each hash
    hash1 = hash1.ljust(64, '0')
    hash2 = hash2.ljust(64, '0')

    # Calculate the Hamming distance
    distance = 0
    for i in range(64):
        if hash1[i]!= hash2[i]:
            distance += 1

    return distance / 64

if __name__ == "__main__":
    hash1 = "01234567890123456789012345678901"
    hash2 = "01010101010101010101010101010101"
    print(f"Hash 1: {hash1}")
    print(f"Hash 2: {hash2}")
    print(f"Hamming distance: {hamming_distance(hash1, hash2)}")