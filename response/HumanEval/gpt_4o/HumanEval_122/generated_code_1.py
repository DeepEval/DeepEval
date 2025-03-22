def method(arr, k):
    # Ensure k does not exceed the length of arr
    k = min(k, len(arr))
    
    # Initialize sum
    total_sum = 0
    
    # Iterate over the first k elements
    for i in range(k):
        # Check if the absolute value of the element has at most two digits
        if -100 < arr[i] < 100:
            total_sum += arr[i]
    
    return total_sum

# Test case
arr = [15, 203, -50, 7, 1000, 99, -123]
k = 5
# Expected output: 15 - 50 + 7 = -28, since 203 and 1000 have more than two digits
print(method(arr, k))  # Output should be -28