def hamming_distance(hash1, hash2):
    # Ensure both hashes are padded to 64 bits if they are shorter
    hash1 = hash1.ljust(64, '0')
    hash2 = hash2.ljust(64, '0')

    # Calculate the Hamming distance
    distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    return float(distance)

if __name__ == "__main__":
    # Example hashes
    hash1 = "1100101011111110"
    hash2 = "1100101011110000"
    
    # Calculate and print the Hamming distance
    print(hamming_distance(hash1, hash2))