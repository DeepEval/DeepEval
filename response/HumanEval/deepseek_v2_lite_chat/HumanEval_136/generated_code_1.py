def method():
    # List of integers
    numbers = [1, 2, 3, 4, -5, -6, -7, -8, 9, 10, -11, 12, -13, -14, 15]
    
    # Initialize variables to store the largest negative and smallest positive
    largest_negative = float('-inf')
    smallest_positive = float('inf')
    
    # Iterate through the list to find the largest negative and smallest positive
    for num in numbers:
        if num < 0:
            largest_negative = max(largest_negative, num)
        elif num > 0:
            smallest_positive = min(smallest_positive, num)
    
    # Return the result as a tuple (largest negative, smallest positive)
    # If there are no negative or positive numbers, return None for both
    return (largest_negative, smallest_positive) if largest_negative != float('-inf') and smallest_positive != float('inf') else None

# Test case to validate the function
def test_method():
    # assert method() == (-5, 2)
    # assert method() == (None, None)
    # assert method() == (None, 1)
    # assert method() == (-11, 2)
    method()

test_method()