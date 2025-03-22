def method():
    # Input: lengths of the three sides of the triangle
    a = 3
    b = 4
    c = 5

    # Sort the sides to ensure c is the hypotenuse
    sides = sorted([a, b, c])

    # Check if the Pythagorean theorem holds
    if sides[2]**2 == sides[0]**2 + sides[1]**2:
        output = True
    else:
        output = False

    return output

# Test case
print(method())  # Expected output: True