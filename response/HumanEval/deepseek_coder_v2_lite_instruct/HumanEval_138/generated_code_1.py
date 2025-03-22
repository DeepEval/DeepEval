def method(n):
    # A number can be expressed as the sum of 4 positive even numbers if and only if it is even and greater than or equal to 8
    if n % 2 == 0 and n >= 8:
        return True
    else:
        return False

# Test case
n = 10
output = method(n)
print(f"The number {n} can be written as the sum of exactly 4 positive even numbers: {output}")