def cycpattern_check(s1, s2):
    """
    Checks if the second word or any of its rotations is a substring in the first word.

    Args:
        s1: The first word.
        s2: The second word.

    Returns:
        True if the second word or any of its rotations is a substring in the first word, False otherwise.
    """

    # Create a list of rotations of s2.
    rotations = []
    for i in range(len(s2)):
        rotations.append(s2[i:] + s2[:i])

    # Check if any of the rotations is a substring in s1.
    for rotation in rotations:
        if rotation in s1:
            return True

    # If no rotations are found in s1, return False.
    return False


# Test case.
s1 = "hello"
s2 = "ell"

print(cycpattern_check(s1, s2))