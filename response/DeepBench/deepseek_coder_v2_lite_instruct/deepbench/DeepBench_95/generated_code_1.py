def hamming_distance(hash1, hash2):
    # Pad the hashes to 64 bits if necessary
    hash1 = hash1.zfill(64)
    hash2 = hash2.zfill(64)
    
    # Calculate the Hamming distance
    distance = 0
    for bit1, bit2 in zip(hash1, hash2):
        if bit1 != bit2:
            distance += 1
    
    return distance

if __name__ == "__main__":
    # Example usage
    hash1 = "1101011010110101101011010110101101011010110101101011010110101101"
    hash2 = "1010101010101010101010101010101010101010101010101010101010101010"
    print("Hamming distance:", hamming_distance(hash1, hash2))