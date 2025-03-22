def method(a):
    # Calculate the cube root of the integer
    cube_root = round(a ** (1.0/3))
    
    # Check if the cube of the cube root is equal to the original integer
    if cube_root ** 3 == a:
        return True
    else:
        return False

# Test case
output = method(27)
print(output)  # Expected output: True

output = method(16)
print(output)  # Expected output: False