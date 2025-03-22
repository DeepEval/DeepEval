def method(arr, k):
    # Initialize the sum
    total_sum = 0
    
    # Iterate over the first k elements of arr
    for i in range(min(k, len(arr))):  # Ensure we don't go out of bounds
        if -100 < arr[i] < 100:  # Check if the element has at most two digits
            total_sum += arr[i]  # Add to the sum
    
    return total_sum

# Test case
arr = [1, 23, 456, -78, 90, 101, -12]
k = 5
output = method(arr, k)
print(output)  # Expected output: 1 + 23 + (-78) + 90 + (-12) = 24