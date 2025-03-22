def pairs_sum_to_zero(nums):
    """
    Returns True if there are two distinct elements in the list that sum to zero, and False otherwise.

    Args:
        nums (list): A list of integers.

    Returns:
        bool: Whether two distinct elements sum to zero.
    """
    seen = set()  # Initialize an empty set to store seen numbers.
    for num in nums:  # Iterate over the list of numbers.
        complement = -num  # Calculate the complement of the current number.
        if complement in seen:  # Check if the complement has been seen before.
            return True  # If the complement has been seen, return True.
        seen.add(num)  # Add the current number to the set of seen numbers.
    return False  # If no pair with a sum of zero is found, return False.

# Test case
print(pairs_sum_to_zero([1, 2, 3, 4, 5]))  # Expected output: False
print(pairs_sum_to_zero([1, -1, 2, 3, 4]))  # Expected output: True
print(pairs_sum_to_zero([0, 0, 0]))  # Expected output: True
print(pairs_sum_to_zero([1, 2, 3, -1, -2]))  # Expected output: True