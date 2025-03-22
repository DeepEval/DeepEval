def is_prime(n):
    """Check if a number is a prime number."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def method(input_string):
    """Return True if the length of the input string is a prime number, False otherwise."""
    string_length = len(input_string)
    return is_prime(string_length)

# Test case
test_string = "hello"
output = method(test_string)
print(output)  # Expected output: True