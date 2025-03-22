def method(a, b, c):
    # Sort the sides to easily identify the longest side
    sides = sorted([a, b, c])
    
    # Assign sides after sorting
    a, b, c = sides
    
    # Check the Pythagorean theorem
    return a**2 + b**2 == c**2

# Test case
output = method(3, 4, 5)
print(output)  # Should print True because 3^2 + 4^2 = 5^2