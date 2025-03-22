import math

def method(a, b, c):
    # Check for a valid triangle
    if (a + b > c) and (a + c > b) and (b + c > a):
        # Calculate the semi-perimeter
        s = (a + b + c) / 2
        # Calculate the area using Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return round(area, 2)
    else:
        return -1

# Test case
output = method(3, 4, 5)
print(output)  # Expected output: 6.00