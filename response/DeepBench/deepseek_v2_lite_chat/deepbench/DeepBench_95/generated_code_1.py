from Crypto.Hash import SHA256

def hamming_distance(hash1, hash2):
    hash1 = hash1.encode()
    hash2 = hash2.encode()

    if len(hash1) % 64 != 0:
        pad_length = 64 - (len(hash1) % 64)
        hash1 += '\0' * pad_length
    if len(hash2) % 64 != 0:
        pad_length = 64 - (len(hash2) % 64)
        hash2 += '\0' * pad_length

    return sum(byte1 ^ byte2 for byte1, byte2 in zip(hash1, hash2)) / len(hash1)

if __name__ == "__main__":
    hash1 = "1b37373331363f78151b7a2b743172696e7977797979"
    hash2 = "686974202c6b696e206f742073616d6520616e6421"
    print(hamming_distance(hash1, hash2))