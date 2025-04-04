import math

def is_prime(num):
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

def method():
    # Test case:
    test_num = 29
    output = is_prime(test_num)
    return output

# Example usage:
# result = method()
# print("The number", test_num, "is prime:", result)