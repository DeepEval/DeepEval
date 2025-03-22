from math import sqrt

def is_prime(num):
    """
    This function takes a number as input and returns True if the number is prime,
    False otherwise.
    """
    if num <= 1:
        return False
    for i in range(2, int(sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def method(str):
    """
    This function takes a string as input and returns True if the string length
    is a prime number, False otherwise.
    """
    length = len(str)
    if is_prime(length):
        output = True
    else:
        output = False
    return output

# Test case
str = "hello"
print(method(str))