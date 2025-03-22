import math

def calculate_area(a, b, c):
    """
    Calculate the area of a triangle given its sides.

    Args:
    a (float): The length of the first side.
    b (float): The length of the second side.
    c (float): The length of the third side.

    Returns:
    float: The area of the triangle rounded to 2 decimal points if the sides form a valid triangle, otherwise -1.
    """
    # Check if the sides form a valid triangle
    if a + b > c and a + c > b and b + c > a:
        # Calculate the semi-perimeter
        s = (a + b + c) / 2
        
        # Calculate the area using Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        
        # Return the area rounded to 2 decimal points
        return round(area, 2)
    else:
        # Return -1 if the sides do not form a valid triangle
        return -1

# Test the function
def method():
    a = 3
    b = 4
    c = 5
    output = calculate_area(a, b, c)
    return output

output = method()
print("The final answer is", output)