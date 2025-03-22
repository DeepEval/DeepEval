def method(lst):
    """
    Given a non-empty list of integers, adds the even elements that are at odd indices.

    Args:
        lst (list): A non-empty list of integers.

    Returns:
        int: The sum of even elements at odd indices in the list.
    """
    # Initialize a variable to store the sum of even elements at odd indices
    output = 0
    
    # Iterate over the list with enumerate to get both index and value
    for i, num in enumerate(lst):
        # Check if the index is odd and the number is even
        if i % 2!= 0 and num % 2 == 0:
            # Add the even number at odd index to the output
            output += num
    
    return output

# Test case
if __name__ == "__main__":
    # Test list with even and odd elements at odd indices
    test_list = [2, 5, 8, 1, 4, 3, 10]
    print(method(test_list))  # Expected output: 26 (8 + 4 + 10)