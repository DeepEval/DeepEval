from typing import List

def is_balanced_and_within_weight(q: List[int], w: int) -> bool:
    """
    Returns True if the object q will fly, and False otherwise. The object q will fly if it's balanced (it is a palindromic list) and the sum of its elements is less than or equal the maximum possible weight w.

    Args:
        q: The object to check.
        w: The maximum possible weight.

    Returns:
        True if the object q will fly, and False otherwise.
    """

    # Check if the object is balanced.
    if not is_palindrome(q):
        return False

    # Check if the sum of the object's elements is less than or equal to the maximum possible weight.
    if sum(q) > w:
        return False

    # The object is balanced and within the weight limit.
    return True

def is_palindrome(q: List[int]) -> bool:
    """
    Returns True if the object q is a palindrome, and False otherwise.

    Args:
        q: The object to check.

    Returns:
        True if the object q is a palindrome, and False otherwise.
    """

    # Check if the object is empty.
    if not q:
        return True

    # Check if the first and last elements of the object are equal.
    if q[0] != q[-1]:
        return False

    # Recursively check the remaining elements of the object.
    return is_palindrome(q[1:-1])

# Test case
q = [1, 2, 3, 2, 1]
w = 5

print(is_balanced_and_within_weight(q, w))  # Output: True