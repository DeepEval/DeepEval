def method():
    # Example list
    lst = [1, 2, 3, 4, 5]
    
    # Check if the list is monotonically increasing
    is_increasing = all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))
    
    # Check if the list is monotonically decreasing
    is_decreasing = all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))
    
    # Return True if either is true
    output = is_increasing or is_decreasing
    return output

# Test case
print(method())  # Expected output: True