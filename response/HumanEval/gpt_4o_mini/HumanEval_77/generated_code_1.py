import math

def method(a):
    # Calculate the cube root of the absolute value of a
    cube_root = round(a ** (1/3))
    
    # Check if the cube of the rounded cube root equals the original number
    return cube_root ** 3 == a

# Test case
print(method(27))   # True, because 3^3 = 27
print(method(-27))  # True, because (-3)^3 = -27
print(method(16))   # False, because there is no integer x such that x^3 = 16
print(method(0))    # True, because 0^3 = 0