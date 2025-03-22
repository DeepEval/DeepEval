import math

def method(a, b, c):
    """
    Validate if the given sides form a right-angled triangle.

    Args:
        a (float): Length of side a.
        b (float): Length of side b.
        c (float): Length of side c.

    Returns:
        bool: True if the sides form a right-angled triangle, False otherwise.
    """
    # Sort the sides in ascending order
    sides = sorted([a, b, c])

    # According to Pythagorean theorem, in a right-angled triangle,
    # the square of the hypotenuse (longest side) is equal to the sum of squares of the other two sides.
    return math.isclose(sides[2]**2, sides[0]**2 + sides[1]**2)


# Test case
if __name__ == "__main__":
    a = 3
    b = 4
    c = 5
    output = method(a, b, c)
    print(f"Output: {output}")