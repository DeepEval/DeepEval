import math

def is_cube(a):
    """
    This function checks if a given integer is a cube of some integer number.

    Args:
        a (int): The input integer to be checked.

    Returns:
        bool: True if the integer is a cube of some integer number, False otherwise.
    """
    cube_root = round(math.pow(a, 1/3))
    return cube_root ** 3 == a


def method():
    a = 27  # Example input
    return is_cube(a)


output = method()
print(output)  # Expected output: True

# Test cases
print(is_cube(27))  # Expected output: True
print(is_cube(8))   # Expected output: False