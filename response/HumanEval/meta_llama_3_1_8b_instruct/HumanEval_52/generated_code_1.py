def method(l, t):
    """
    Returns True if all numbers in the list l are below threshold t.

    Args:
        l (list): A list of numbers.
        t (int): The threshold value.

    Returns:
        bool: True if all numbers in the list are below the threshold, False otherwise.
    """
    return all(num < t for num in l)


# Test case
l = [1, 2, 3, 4, 5]
t = 4
output = method(l, t)
print(output)  # Expected output: True