def hamming_distance(hash1: str, hash2: str) -> float:
    max_length = 64
    
    hash1 = hash1.zfill(max_length)
    hash2 = hash2.zfill(max_length)

    distance = sum(el1 != el2 for el1, el2 in zip(hash1, hash2))
    return float(distance)

if __name__ == "__main__":
    hash1 = "1101010101010101010101010101010101010101010101010101010101010101"
    hash2 = "1101010101010101000000000000000000000000000000000000000000000000"
    
    result = hamming_distance(hash1, hash2)
    print("Hamming Distance:", result)