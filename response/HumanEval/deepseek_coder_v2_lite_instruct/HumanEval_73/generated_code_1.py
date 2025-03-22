def method():
    # Example array
    arr = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    
    # Initialize the number of changes needed
    changes = 0
    
    # Use two pointers to compare elements from start and end
    left, right = 0, len(arr) - 1
    
    while left < right:
        if arr[left] != arr[right]:
            # Increment the change count
            changes += 1
        # Move the pointers towards the center
        left += 1
        right -= 1
    
    # Return the final output
    output = changes
    return output

# Test case
print(method())  # Expected output: 0 (the array is already palindromic)