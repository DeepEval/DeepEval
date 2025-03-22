def method(n):
    # Step 1: Create the array a based on the given formula
    a = [i * i - i + 1 for i in range(1, n + 1)]

    # Step 2: Count occurrences of each remainder when divided by 3
    count = [0] * 3
    for value in a:
        count[value % 3] += 1

    # Step 3: Calculate the number of valid triples
    result = 0

    # Case 1: All three elements have remainder 0
    if count[0] >= 3:
        result += (count[0] * (count[0] - 1) * (count[0] - 2)) // 6

    # Case 2: All three elements have remainder 1
    if count[1] >= 3:
        result += (count[1] * (count[1] - 1) * (count[1] - 2)) // 6

    # Case 3: All three elements have remainder 2
    if count[2] >= 3:
        result += (count[2] * (count[2] - 1) * (count[2] - 2)) // 6

    # Case 4: One element from each remainder
    result += count[0] * count[1] * count[2]

    return result

# Test case for validation
n = 5
output = method(n)
print(f"Number of valid triples for n={n}: {output}")