import re

def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def count_prime_digits(hex_str):
    prime_digits = {'2', '3', '5', '7', 'B', 'D'}
    count = 0

    # Replace hex digits with their decimal equivalents
    hex_str = re.sub(r'[A-F]', lambda m: str(int(m.group(), 16)), hex_str)

    # Check if each decimal digit is prime
    for digit in hex_str:
        if digit in prime_digits and is_prime(int(digit)):
            count += 1

    return count

# Test case
hex_str = 'AB57C'
print(count_prime_digits(hex_str))  # Output: 3