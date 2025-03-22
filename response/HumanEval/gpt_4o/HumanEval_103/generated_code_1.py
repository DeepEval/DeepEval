def method(n, m):
    if n > m:
        return -1

    # Calculate the sum of integers from n to m
    total_sum = sum(range(n, m + 1))
    # Calculate the number of integers in the range
    count = m - n + 1
    # Calculate the average and round it to the nearest integer
    average = round(total_sum / count)
    # Convert the average to binary and return it as a string
    output = bin(average)[2:]  # [2:] to remove the '0b' prefix

    return output

# Test case
n = 4
m = 8
print(method(n, m))  # Expected output: '110' (6 in binary, since average of [4, 5, 6, 7, 8] is 6)