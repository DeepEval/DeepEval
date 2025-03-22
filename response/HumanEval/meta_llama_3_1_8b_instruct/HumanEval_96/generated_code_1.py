import math

def is_prime(num):
    """
    Checks if a number is prime.

    Args:
    num (int): The number to check.

    Returns:
    bool: True if the number is prime, False otherwise.
    """
    if num <= 1:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def method(n):
    """
    Returns a list of the first n prime numbers less than n.

    Args:
    n (int): The upper limit.

    Returns:
    list: A list of the first n prime numbers less than n.
    """
    output = []
    num = 2
    while len(output) < n:
        
        # 修改
        if num >= n:
            break

        if num < n and is_prime(num):
            output.append(num)
        num += 1
    return output

# Test case
n = 10
print(method(n))  # Output: [2, 3, 5, 7]