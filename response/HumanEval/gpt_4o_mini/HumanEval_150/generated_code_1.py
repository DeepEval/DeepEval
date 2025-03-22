def is_prime(num):
    """Check if a number is prime."""
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def method(n, x, y):
    """Return x if n is a prime number, else return y."""
    if is_prime(n):
        return x
    else:
        return y

# Test case
n = 7  # A prime number
x = "Prime"
y = "Not Prime"
output = method(n, x, y)
print(output)  # Expected output: "Prime"

# Another test case
n = 4  # Not a prime number
output = method(n, x, y)
print(output)  # Expected output: "Not Prime"