import math

def is_prime(num):
    """Check if a number is prime."""
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

def prime_fib(n):
    """Return the n-th Fibonacci number that is also prime."""
    if n < 1:
        return None

    # Initialize Fibonacci sequence
    fib1, fib2 = 0, 1
    count = 0  # To count how many prime Fibonacci numbers have been found

    while True:
        fib_next = fib1 + fib2
        if is_prime(fib_next):
            count += 1
            if count == n:
                return fib_next
        fib1, fib2 = fib2, fib_next

def method():
    # Example test case: Find the 5th prime Fibonacci number
    output = prime_fib(5)
    return output

# Test case
print(method())  # This will print the 5th prime Fibonacci number