def method(a):
    # Calculate the cube root of a and round it to the nearest integer
    cube_root = round(a ** (1/3))
    
    # Check if the cube of this integer is equal to a
    if cube_root ** 3 == a:
        return True
    else:
        return False

# Test case
test_case = 27  # 3^3 = 27
print(method(test_case))  # Output should be True

# Additional test cases
print(method(64))  # 4^3 = 64, should return True
print(method(16))  # 16 is not a perfect cube, should return False
print(method(-27)) # (-3)^3 = -27, should return True
print(method(0))   # 0^3 = 0, should return True