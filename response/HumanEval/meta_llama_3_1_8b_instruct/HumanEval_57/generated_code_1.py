def is_monotonic(lst):
    """
    Returns True if the elements in the list are monotonically increasing or decreasing.
    
    Args:
    lst (list): A list of elements to be checked.
    
    Returns:
    bool: True if the elements are monotonically increasing or decreasing, False otherwise.
    """
    
    # Check if the list is empty
    if len(lst) == 0:
        return True
    
    # Check if the list has only one element
    if len(lst) == 1:
        return True
    
    # Check if the list is monotonically increasing
    increasing = all(lst[i] <= lst[i+1] for i in range(len(lst)-1))
    
    # Check if the list is monotonically decreasing
    decreasing = all(lst[i] >= lst[i+1] for i in range(len(lst)-1))
    
    # Return True if the list is either monotonically increasing or decreasing
    return increasing or decreasing


# Test case
print(is_monotonic([1, 2, 3, 4, 5]))  # Expected output: True
print(is_monotonic([5, 4, 3, 2, 1]))  # Expected output: True
print(is_monotonic([1, 3, 2, 4, 5]))  # Expected output: False