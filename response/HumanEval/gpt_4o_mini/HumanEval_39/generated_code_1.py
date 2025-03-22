def is_prime(num):
    """Check if a number is prime."""
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while (i * i) <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

def fibonacci_prime(n):
    """Return the n-th Fibonacci number that is also prime."""
    fib1, fib2 = 0, 1
    prime_fibs = []
    
    while len(prime_fibs) < n:
        fib1, fib2 = fib2, fib1 + fib2  # Generate the next Fibonacci number
        if is_prime(fib2):
            prime_fibs.append(fib2)
    
    return prime_fibs[n - 1]

def method():
    n = 5  # Example for the 5th prime Fibonacci number
    output = fibonacci_prime(n)
    return output

# Test case for validation
if __name__ == "__main__":
    result = method()
    print(f"The 5th Fibonacci prime number is: {result}")  # Expected output: 5