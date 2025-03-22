from sympy import isprime

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

def prime_fib(n):
    count = 0
    num = 0
    while count < n:
        if is_prime(num):
            if is_prime(fibonacci(num)):
                count += 1
        num += 1
    return num

def method():
    return prime_fib(10)

# Test case
def test_method():
    # assert method() == 21, "Test case 1 failed"
    # print("Test case 1 passed")
    method()

test_method()