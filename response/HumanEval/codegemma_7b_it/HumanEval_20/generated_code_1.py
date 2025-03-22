import math

def method(lst):
    """
    Finds the two closest numbers in a list and returns them in order (smaller, larger).

    Args:
        lst: A list of numbers.

    Returns:
        A tuple containing the two closest numbers in order (smaller, larger).
    """

    # Sort the list in ascending order.
    lst.sort()

    # Initialize the closest pair and the minimum difference.
    closest_pair = (None, None)
    min_diff = math.inf

    # Iterate over the list, starting from the second element.
    for i in range(1, len(lst)):
        # Calculate the difference between the current number and the previous number.
        diff = abs(lst[i] - lst[i - 1])

        # If the difference is smaller than the current minimum difference, update the closest pair and minimum difference.
        if diff < min_diff:
            min_diff = diff
            closest_pair = (lst[i - 1], lst[i])

    # Return the closest pair.
    return closest_pair


# Test case.
lst = [1, 4, 2, 8, 5, 3, 6]
closest_pair = method(lst)
print(closest_pair)  # Output: (2, 3)