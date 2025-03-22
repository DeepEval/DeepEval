def next_smallest(nums):
    """
    Returns the 2nd smallest element of a list of integers.
    If the list has less than 2 elements, returns None.
    
    Parameters:
    nums (list): A list of integers.
    
    Returns:
    int or None: The 2nd smallest element of the list, or None if the list has less than 2 elements.
    """
    if len(nums) < 2:  # If the list has less than 2 elements, return None
        return None
    
    nums.sort()  # Sort the list in ascending order
    if nums[0] == nums[1]:  # If the first two elements are equal, return None
        return None
    
    return nums[1]  # Return the second smallest element


# Test cases
print(next_smallest([1, 2, 3, 4, 5]))  # Expected output: 2
print(next_smallest([5, 1, 4, 3, 2]))  # Expected output: 2
print(next_smallest([]))  # Expected output: None
print(next_smallest([1, 1]))  # Expected output: None