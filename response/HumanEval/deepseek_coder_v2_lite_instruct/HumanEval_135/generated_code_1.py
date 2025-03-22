def method():
    # Example array
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Initialize the output to -1
    output = -1
    
    # Iterate through the array
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            output = i
    
    return output

# Test case
print(method())  # Expected output: 8