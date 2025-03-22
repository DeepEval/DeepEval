def method(n, m):
    if n > m:
        return -1
    
    # Calculate the sum of integers from n to m
    total_sum = sum(range(n, m + 1))
    
    # Calculate the average
    average = total_sum / (m - n + 1)
    
    # Round the average to the nearest integer
    rounded_average = round(average)
    
    # Convert the rounded average to binary
    binary_output = bin(rounded_average)[2:]  # [2:] to remove the '0b' prefix
    
    return binary_output

# Test case
n = 3
m = 7
print(method(n, m))  # Expected output: '100'