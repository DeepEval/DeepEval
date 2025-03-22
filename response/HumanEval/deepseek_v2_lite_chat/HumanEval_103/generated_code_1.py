import math

def method():
    # Check if n is greater than m
    if n > m:
        return -1
    
    # Calculate the average
    average = (n + m) / 2
    
    # Round the average to the nearest integer
    rounded_average = round(average)
    
    # Convert the rounded average to binary
    binary_average = bin(rounded_average)[2:]  # [2:] to remove the '0b' prefix
    
    return binary_average

# Test case
n = 5
m = 10
result = method()
print("Test Case:", n, "through", m)
print("Result:", result)