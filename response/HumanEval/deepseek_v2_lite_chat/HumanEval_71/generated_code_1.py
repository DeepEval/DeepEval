import math

def method():
    # Function to check if a triangle is valid
    def is_valid_triangle(a, b, c):
        return (a + b > c) and (a + c > b) and (b + c > a)

    # Function to calculate the area using the Shoelace formula
    def triangle_area(a, b, c):
        # Calculate the semi-perimeter
        s = (a + b + c) / 2
        # Calculate the area using the Shoelace formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return area

    # Function to validate the input
    def validate_input(a, b, c):
        if not is_valid_triangle(a, b, c):
            return -1
        else:
            return round(triangle_area(a, b, c), 2)

    # Input validation
    # a = float(input("Enter the length of the first side: "))
    # b = float(input("Enter the length of the second side: "))
    # c = float(input("Enter the length of the third side: "))

    a = 3
    b = 4
    c = 5

    # Validate the input and return the result
    return validate_input(a, b, c)

# Test the function
print(method())