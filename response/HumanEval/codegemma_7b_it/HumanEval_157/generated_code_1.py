import math

def method():
    # Get the lengths of the three sides of the triangle from the user.
    # side1 = float(input("Enter the length of side 1: "))
    # side2 = float(input("Enter the length of side 2: "))
    # side3 = float(input("Enter the length of side 3: "))
    side1 = 3
    side2 = 4
    side3 = 5

    # Check if the triangle is a right-angled triangle.
    if (side1**2 + side2**2 == side3**2) or (side2**2 + side3**2 == side1**2) or (side3**2 + side1**2 == side2**2):
        output = True
    else:
        output = False

    return output

# Test case for validation.
test_case = method()
print(test_case)