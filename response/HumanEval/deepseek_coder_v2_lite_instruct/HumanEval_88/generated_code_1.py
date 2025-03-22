def method():
    # Example array
    arr = [3, 1, 2, 4, 5]
    
    # Copy the array to avoid changing the original array
    arr_copy = arr.copy()
    
    # Calculate the sum of the first and last elements
    sum_first_last = arr_copy[0] + arr_copy[-1]
    
    # Sort the array based on the sum being odd or even
    if sum_first_last % 2 == 0:
        # If the sum is even, sort in descending order
        arr_copy.sort(reverse=True)
    else:
        # If the sum is odd, sort in ascending order
        arr_copy.sort()
    
    return arr_copy

# Test case
output = method()
print(output)  # Expected output: [1, 2, 3, 4, 5] if sum_first_last is odd, [5, 4, 3, 2, 1] if sum_first_last is even