import math

def method(a, b, c):
    # Check if the sides form a valid triangle
    if a + b > c and a + c > b and b + c > a:
        # Calculate the semi-perimeter
        s = (a + b + c) / 2
        # Calculate the area using Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        # Round the area to 2 decimal places
        return round(area, 2)
    else:
        # Return -1 if it's not a valid triangle
        return -1

# Test case
print(method(3, 4, 5))  # Expected output: 6.00 (3-4-5 is a well-known right triangle)
print(method(1, 1, 2))  # Expected output: -1 (Not a valid triangle)