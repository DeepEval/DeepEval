def method(n):
    # Check if n is even and greater than or equal to 8
    if n >= 8 and n % 2 == 0:
        return True
    else:
        return False

# Test case
print(method(12))  # True, because 12 can be written as 2 + 2 + 4 + 4
print(method(10))  # True, because 10 can be written as 2 + 2 + 2 + 4
print(method(7))   # False, because 7 is not even
print(method(6))   # False, because 6 is less than 8