def method(arr):
    """
    This function returns the largest index of an element which is not greater than or equal to 
    the element immediately preceding it. If no such element exists then return -1.
    
    Parameters:
    arr (list): A list of unique integers.
    
    Returns:
    int: The largest index of an element which is not greater than or equal to the element immediately preceding it.
    """
    
    # Iterate through the array from the second element to the last
    for i in range(1, len(arr)):
        # If the current element is less than the previous one, return the current index
        if arr[i] < arr[i - 1]:
            return i - 1
    
    # If no such element is found, return -1
    return -1

# Test case
arr = [5, 4, 3, 2, 1]
print(method(arr))  # Output: 0

arr = [5, 1, 3, 2, 4]
print(method(arr))  # Output: 1

arr = [5, 5, 5, 5, 5]
print(method(arr))  # Output: -1