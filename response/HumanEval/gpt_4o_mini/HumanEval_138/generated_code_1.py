def method(n):
    # Check if n is even and at least 8
    if n >= 8 and n % 2 == 0:
        return True
    return False

# Test case
n = 12  # 12 can be expressed as 2 + 2 + 4 + 4
output = method(n)
print(f"Can {n} be expressed as the sum of 4 positive even numbers? {output}")

n = 7  # 7 cannot be expressed as the sum of 4 positive even numbers
output = method(n)
print(f"Can {n} be expressed as the sum of 4 positive even numbers? {output}")

n = 10  # 10 can be expressed as 2 + 2 + 2 + 4
output = method(n)
print(f"Can {n} be expressed as the sum of 4 positive even numbers? {output}")

n = 8  # 8 can be expressed as 2 + 2 + 2 + 2
output = method(n)
print(f"Can {n} be expressed as the sum of 4 positive even numbers? {output}")