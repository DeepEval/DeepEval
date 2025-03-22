def double_the_difference(numbers):
    """
    This function calculates the sum of squares of odd numbers in a list.
    
    Args:
        numbers (list): A list of integers.
    
    Returns:
        int: The sum of squares of odd numbers in the list.
    """
    # Initialize the sum to 0
    total_sum = 0
    
    # Iterate over each number in the list
    for num in numbers:
        # Check if the number is an integer and not negative
        if isinstance(num, int) and num >= 0:
            # Check if the number is odd
            if num % 2!= 0:
                # Add the square of the number to the total sum
                total_sum += num ** 2
    
    # Return the total sum
    return total_sum

# Test cases
print(double_the_difference([1, 3, 2, 0]))  # Expected output: 10
print(double_the_difference([-1, -2, 0]))  # Expected output: 0
print(double_the_difference([9, -2]))  # Expected output: 81
print(double_the_difference([0]))  # Expected output: 0
print(double_the_difference([]))  # Expected output: 0