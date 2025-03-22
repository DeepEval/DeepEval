def method(arr):
    """
    This function checks if it's possible to get an array sorted in non-decreasing order by performing the right shift operation.
    
    Parameters:
    arr (list): A list of unique integers in random order.
    
    Returns:
    bool: True if it's possible to get an array sorted in non-decreasing order, False otherwise.
    """
    
    # Check if the array is empty
    if not arr:
        return True
    
    # Sort the array in ascending order
    sorted_arr = sorted(arr)
    
    # Initialize the index for the sorted array
    sorted_index = 0
    
    # Iterate over the original array
    for num in arr:
        # If the current number is greater than the corresponding number in the sorted array, 
        # it means the array cannot be sorted by right shifting
        if num!= sorted_arr[sorted_index]:
            return False
        # Move to the next number in the sorted array
        sorted_index += 1
    
    # If the loop completes without returning False, it means the array can be sorted by right shifting
    return True


# Test case
arr = [4, 2, 7, 1, 3]
output = method(arr)
print("Is it possible to get the array sorted in non-decreasing order by right shifting?", output)