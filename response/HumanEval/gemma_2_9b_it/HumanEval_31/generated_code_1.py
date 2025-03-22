def method(num):
    """
    Returns True if a given number is prime, and False otherwise.
    """
    # Prime numbers are greater than 1
    if num <= 1:
        return False
    # Check if num is divisible by any number from 2 to the square root of num
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# Test case
number = 17
output = method(number)
print(f"Is {number} a prime number? {output}")