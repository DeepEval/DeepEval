def method(arr):
    # Initialize the count of changes needed
    changes = 0

    # Use two pointers to compare elements from the start and end
    left = 0
    right = len(arr) - 1

    # Iterate until the two pointers meet or cross
    while left < right:
        # If elements are not equal, increment the changes counter
        if arr[left] != arr[right]:
            changes += 1
        
        # Move the pointers towards the center
        left += 1
        right -= 1

    # Return the total number of changes needed
    return changes

# Test case
arr = [1, 2, 3, 4, 3, 2, 1]  # This array is already palindromic
output = method(arr)
print(f"Minimum number of changes needed: {output}")  # Expected output: 0

# Additional test case
arr = [1, 2, 3, 4, 5, 6, 7]  # Needs changes to become palindromic
output = method(arr)
print(f"Minimum number of changes needed: {output}")  # Expected output: 3