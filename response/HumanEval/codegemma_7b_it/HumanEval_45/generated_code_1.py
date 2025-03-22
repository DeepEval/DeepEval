from math import sqrt

def method():

    # Get the length of the side and high from the user
    # side = float(input("Enter the length of the side: "))
    # high = float(input("Enter the high of the triangle: "))
    side = 5
    high = 10

    # Calculate the area of the triangle
    area = 0.5 * side * high

    # Return the area of the triangle
    return area

# Test case
test_case = (5, 10)

# Call the method() function and get the area of the triangle
area = method()

# Print the area of the triangle
print("The area of the triangle is: ", area)