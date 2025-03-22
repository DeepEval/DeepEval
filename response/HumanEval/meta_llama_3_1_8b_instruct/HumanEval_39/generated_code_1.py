def is_prime(n):
    """
    Helper function to check if a number is prime.
    
    Args:
    n (int): The number to be checked.
    
    Returns:
    bool: True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    max_divisor = int(n**0.5) + 1
    for d in range(3, max_divisor, 2):
        if n % d == 0:
            return False
    return True


def is_fibonacci(n):
    """
    Helper function to check if a number is a Fibonacci number.
    
    Args:
    n (int): The number to be checked.
    
    Returns:
    bool: True if the number is a Fibonacci number, False otherwise.
    """
    x = 0
    y = 1
    while y < n:
        z = x + y
        x = y
        y = z
    return y == n


def prime_fib(n):
    """
    Returns the n-th number that is a Fibonacci number and it's also prime.
    
    Args:
    n (int): The position of the number in the sequence.
    
    Returns:
    int: The n-th number that is a Fibonacci number and it's also prime.
    """
    num = 1
    count = 0
    while True:
        if is_fibonacci(num) and is_prime(num):
            count += 1
            if count == n:
                return num
        num += 1


# Test case
print(prime_fib(3))  # Output: 89