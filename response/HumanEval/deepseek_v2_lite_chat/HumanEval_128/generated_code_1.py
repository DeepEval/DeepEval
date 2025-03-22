def method(arr):
    if not arr:  # Check if the array is empty
        return None

    # Initialize the sum with 0
    output = 0

    # Iterate through the array
    for num in arr:
        # Calculate the sign of the number (1 for positive, -1 for negative, 0 for zero)
        sign = -1 if num < 0 else 1
        # Add the magnitude of the number to the sum
        output += abs(num) * sign

    return output

# Test case
print(method([1, -2, 3, -4]))  # Expected output: 10
print(method([]))  # Expected output: None
print(method([1, -3, 5, -7, 9]))  # Expected output: 33
print(method([0, 0, 0, 0]))  # Expected output: 0
print(method([1, -2, 0, -3, 4]))  # Expected output: 10