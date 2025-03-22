def is_palindrome(lst):
    """
    Checks if a list is a palindrome.
    
    Args:
    lst (list): The list to be checked.
    
    Returns:
    bool: True if the list is a palindrome, False otherwise.
    """
    return lst == lst[::-1]


def method(lst, w):
    """
    Checks if an object will fly based on whether it's balanced and its weight is within the limit.
    
    Args:
    lst (list): The object's characteristics.
    w (int): The maximum possible weight.
    
    Returns:
    bool: True if the object will fly, False otherwise.
    """
    # Check if the list is a palindrome
    is_balanced = is_palindrome(lst)
    
    # Calculate the sum of the list elements
    weight = sum(lst)
    
    # Check if the weight is within the limit
    is_light_enough = weight <= w
    
    # The object will fly if it's balanced and light enough
    return is_balanced and is_light_enough


# Test case
lst = [1, 2, 3, 2, 1]
w = 10
output = method(lst, w)
print("Will the object fly?", output)

# Expected output: True