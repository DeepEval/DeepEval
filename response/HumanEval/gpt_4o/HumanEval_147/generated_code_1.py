def method(n):
    # Calculate the remainder counts of elements in a modulo 3
    remainder_count = [0, 0, 0]

    for i in range(1, n + 1):
        # Calculate a[i] = i * i - i + 1
        a_i = i * i - i + 1
        # Increment the count of the remainder
        remainder_count[a_i % 3] += 1

    # Unpack the counts for easier access
    count0, count1, count2 = remainder_count

    # Calculate number of valid triples
    # Triple can be formed in the following cases:
    # Case 1: All three have the same remainder
    triples_count = (
        (count0 * (count0 - 1) * (count0 - 2)) // 6 +  # choose 3 from count0
        (count1 * (count1 - 1) * (count1 - 2)) // 6 +  # choose 3 from count1
        (count2 * (count2 - 1) * (count2 - 2)) // 6    # choose 3 from count2
    )

    # Case 2: One from each remainder
    triples_count += count0 * count1 * count2

    return triples_count

# Test case for validation
n = 5
print(method(n))  # Expected output: 1