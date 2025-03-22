import math

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def method(x, y, n):
    if is_prime(n):
        return x
    else:
        return y

# Test case
x = 10
y = 20
n = 7
output = method(x, y, n)
print(output)