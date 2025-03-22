import math

def method():
    # Example input: a, b, c are the lengths of the three sides of the triangle
    a, b, c = 3, 4, 5
    
    # Check if the sides form a valid triangle
    if a + b > c and a + c > b and b + c > a:
        # Calculate the semi-perimeter
        s = (a + b + c) / 2
        # Calculate the area using Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        # Round the area to 2 decimal points
        return round(area, 2)
    else:
        # Return -1 if the sides do not form a valid triangle
        return -1

# Test case
output = method()
print(output)  # Expected output: 6.0