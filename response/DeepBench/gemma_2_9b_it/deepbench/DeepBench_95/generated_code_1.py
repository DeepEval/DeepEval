def hamming_distance(hash1, hash2):
  hash1 = hash1.zfill(64)
  hash2 = hash2.zfill(64)
  return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

if __name__ == "__main__":
  hash1 = "10110101010101010101010101010101"
  hash2 = "10010101010101010101010101010101"
  distance = hamming_distance(hash1, hash2)
  print(f"Hamming distance: {distance}")