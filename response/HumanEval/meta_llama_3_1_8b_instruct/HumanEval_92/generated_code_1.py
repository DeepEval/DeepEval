# Import necessary packages
import math

# Define the function to check if one number is equal to the sum of the other two
def method(num1, num2, num3):
    """
    This function checks if one number is equal to the sum of the other two.
    
    Args:
        num1 (int): The first number.
        num2 (int): The second number.
        num3 (int): The third number.
    
    Returns:
        bool: True if one number is equal to the sum of the other two, False otherwise.
    """
    
    # Check if all numbers are integers
    if not (isinstance(num1, int) and isinstance(num2, int) and isinstance(num3, int)):
        return False
    
    # Check if one number is equal to the sum of the other two
    if (num1 == num2 + num3) or (num2 == num1 + num3) or (num3 == num1 + num2):
        return True
    
    # If none of the above conditions are met, return False
    return False

# Test case for validation
print(method(1, 2, 3))  # Should return False
print(method(2, 2, 3))  # Should return True
print(method(5, 3, 7))  # Should return False
print(method(3, 5, 2))  # Should return True
print(method(3, 3, 3))  # Should return True
print(method(5, 2, 7))  # Should return False
print(method(5, 3.5, 7))  # Should return False