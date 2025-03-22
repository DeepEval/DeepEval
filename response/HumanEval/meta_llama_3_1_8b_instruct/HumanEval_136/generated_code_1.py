def method(lst):
    """
    Returns a tuple (a, b) where 'a' is the largest of negative integers and 'b' is the smallest of positive integers in the list.
    If there is no negative or positive integers, returns them as None.
    
    Parameters:
    lst (list): A list of integers.
    
    Returns:
    tuple: A tuple containing the largest negative integer and the smallest positive integer.
    """
    
    # Initialize variables to store the result
    max_negative = None
    min_positive = None
    
    # Iterate over the list to find the largest negative and smallest positive integers
    for num in lst:
        if num < 0:
            # Update max_negative if num is larger than current max_negative
            if max_negative is None or num > max_negative:
                max_negative = num
        elif num > 0:
            # Update min_positive if num is smaller than current min_positive
            if min_positive is None or num < min_positive:
                min_positive = num
                
    # Return the result as a tuple
    return (max_negative, min_positive)


# Test case
lst = [-5, -2, -1, 0, 1, 2, 3, 10]
print(method(lst))  # Output: (-1, 1)

# Test case with no negative integers
lst = [5, 2, 1, 0, 10]
print(method(lst))  # Output: (None, 1)

# Test case with no positive integers
lst = [-5, -2, -1, 0]
print(method(lst))  # Output: (-1, None)

# Test case with empty list
lst = []
print(method(lst))  # Output: (None, None)