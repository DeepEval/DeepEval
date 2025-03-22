def method(n, m):
    # Step 1: Check if n is greater than m
    if n > m:
        return -1

    # Step 2: Calculate the sum and count of integers from n to m
    total_sum = sum(range(n, m + 1))
    count = m - n + 1

    # Step 3: Calculate the average and round it to the nearest integer
    average = total_sum / count
    rounded_average = round(average)

    # Step 4: Convert the rounded average to binary
    binary_representation = bin(rounded_average)[2:]  # [2:] to strip the '0b' prefix

    return binary_representation

# Test case for validation
if __name__ == "__main__":
    n = 3
    m = 7
    output = method(n, m)
    print(f"Binary representation of the average from {n} to {m}: {output}")  # Expected output: '101'