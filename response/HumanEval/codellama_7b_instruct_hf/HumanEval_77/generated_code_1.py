import math

def is_cube(a):
    """
    Checks if a number is a cube of some integer number.

    Args:
        a: The integer to check.

    Returns:
        True if a is a cube of some integer number, False otherwise.
    """

    # Calculate the cube root of a.
    cube_root = math.pow(a, 1/3)

    # Check if the cube root is an integer.
    if cube_root.is_integer():
        return True

    # a is not a cube of some integer number.
    return False


# Test case.
a = 27
result = is_cube(a)

print(result)