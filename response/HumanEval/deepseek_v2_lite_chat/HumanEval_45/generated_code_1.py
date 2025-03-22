import math

def method(a, h):
    # Calculate the area of the triangle
    area = 0.5 * a * h
    return area

# Test case
base_length = 5
height = 10
result = method(base_length, height)

# Print the result
print("The area of the triangle is:", result)