def method(a, b, c):
    # Sort the sides to ensure the largest is the potential hypotenuse
    sides = sorted([a, b, c])
    
    # Check if the square of the largest side equals the sum of the squares of the other two sides
    is_right_angled = sides[0]**2 + sides[1]**2 == sides[2]**2
    
    return is_right_angled

# Test case
print(method(3, 4, 5))  # Should return True, as 3^2 + 4^2 = 5^2
print(method(5, 12, 13))  # Should return True, as 5^2 + 12^2 = 13^2
print(method(1, 1, 1))  # Should return False, as 1^2 + 1^2 != 1^2
print(method(6, 8, 10))  # Should return True, as 6^2 + 8^2 = 10^2
print(method(2, 3, 4))  # Should return False, as 2^2 + 3^2 != 4^2