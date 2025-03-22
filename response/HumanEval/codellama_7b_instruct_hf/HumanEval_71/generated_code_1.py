from math import sqrt

def method():
    # Get the lengths of the three sides from the user.
    # a = float(input("Enter the length of side a: "))
    # b = float(input("Enter the length of side b: "))
    # c = float(input("Enter the length of side c: "))

    # 修改
    a = 3
    b = 4
    c = 5

    # Check if the three sides form a valid triangle.
    if a + b > c and a + c > b and b + c > a:
        # Calculate the area of the triangle.
        s = (a + b + c) / 2
        area = sqrt(s * (s - a) * (s - b) * (s - c))

        # Return the area of the triangle rounded to 2 decimal points.
        return round(area, 2)

    else:
        # Return -1 if the three sides do not form a valid triangle.
        return -1

# Test case for validation.
print(method())