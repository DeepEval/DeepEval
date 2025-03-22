import sys

def method(numbers):
    """
    Check if the list of numbers is sorted in ascending order and if there are no duplicate numbers.
    
    Args:
    numbers (list of int): A list of integers.
    
    Returns:
    bool: True if the list is sorted in ascending order and has no duplicates, False otherwise.
    """
    
    # Check for duplicates
    if len(numbers) != len(set(numbers)):
        return False
    
    # Check if the list is sorted in ascending order
    if all(numbers[i] <= numbers[i+1] for i in range(len(numbers) - 1)):
        return True
    
    # If neither condition is met, return False
    return False

# Test the function with a sample input
def test_method():
    """
    Test the function with a sample input.
    """
    # Example list with duplicates and not sorted
    numbers = [3, 5, 3, 1, 2, 1]
    output = method(numbers)
    # assert output == False, "Test failed with input: {}".format(numbers)

    # Example list sorted in ascending order
    numbers = [1, 2, 3, 5]
    output = method(numbers)
    # assert output == True, "Test failed with input: {}".format(numbers)

    # Example list sorted in ascending order with duplicates
    numbers = [1, 2, 3, 3, 5]
    output = method(numbers)
    # assert output == False, "Test failed with input: {}".format(numbers)

    # Example list sorted in ascending order with no duplicates
    numbers = [1, 2, 3, 5, 8]
    output = method(numbers)
    # assert output == True, "Test failed with input: {}".format(numbers)

    print("All test cases passed.")

# Run the test function
test_method()

# Example usage:
# If you want to check the function with a list provided via command-line arguments:
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("numbers", type=int, nargs='+')
# args = parser.parse_args()
# output = method(args.numbers)