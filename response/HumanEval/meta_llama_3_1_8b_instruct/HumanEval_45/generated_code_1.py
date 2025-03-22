import math

def calculate_triangle_area(base, height):
    """
    Calculate the area of a triangle given the base and height.

    Args:
    base (float): The length of a side of the triangle.
    height (float): The perpendicular distance from the base to the opposite vertex.

    Returns:
    float: The area of the triangle.
    """
    if base <= 0 or height <= 0:
        raise ValueError("Base and height must be positive numbers.")
    area = 0.5 * base * height
    return area

def method():
    # Example usage
    base = 5  # length of a side
    height = 6  # height of the triangle
    area = calculate_triangle_area(base, height)
    return area

# Test case
output = method()
print("The area of the triangle is:", output)