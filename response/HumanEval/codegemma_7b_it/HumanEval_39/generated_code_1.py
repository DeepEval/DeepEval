import math

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def prime_fib(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])

    prime_fibs = []
    for num in fib:
        if is_prime(num):
            prime_fibs.append(num)

    return prime_fibs[n-1]

output = prime_fib(10)
print(output)

# Test case
print(prime_fib(6))