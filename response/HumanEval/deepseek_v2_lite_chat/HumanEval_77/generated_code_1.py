import math

def is_perfect_cube(n):
    # Calculate the cube root of n
    cube_root = round(n ** (1/3))
    
    # Check if the cube root is an integer
    if cube_root ** 3 == n and cube_root != 0:
        return True
    else:
        return False

def method(a):
    # Check if the input is a perfect cube
    if is_perfect_cube(a):
        return True
    else:
        return False

# Test case to validate the function
test_case = 27
result = method(test_case)
print(f"Is {test_case} a perfect cube? Result: {result}")