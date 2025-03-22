def method():
    """
    Given a list of positive integers, return a sorted list of all elements 
    that don't have any even digits.
    
    Args:
    None
    
    Returns:
    list: A sorted list of integers without any even digits.
    """
    
    # Function to check if a number has any even digits
    def has_even_digit(num):
        return any(int(digit) % 2 == 0 for digit in str(num))
    
    # Input list of positive integers
    x = [12, 345, 2, 6, 7896, 1234, 56, 78]
    
    # Filter out numbers with even digits and sort the list
    output = sorted([num for num in x if not has_even_digit(num)])
    
    return output


# Test case
print(method())