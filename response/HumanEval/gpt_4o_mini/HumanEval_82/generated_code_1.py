def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def method(input_string):
    """Return True if the length of the string is a prime number, otherwise False."""
    length = len(input_string)
    return is_prime(length)

# Test case for validation
if __name__ == "__main__":
    test_string = "hello"  # Length is 5, which is prime
    output = method(test_string)
    print(f'The length of "{test_string}" is a prime number: {output}')  # Expected output: True