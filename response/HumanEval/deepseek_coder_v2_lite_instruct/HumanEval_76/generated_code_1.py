import math

def method():
    # Example test case
    x = 8
    n = 2
    output = is_simple_power(x, n)
    return output

def is_simple_power(x, n):
    if x == 1:
        return True  # Any number to the power of 0 is 1
    if n == 1:
        return False  # 1 to any power is not interesting, and we don't handle infinity cases
    power = math.log(x, n)
    return power.is_integer()

# Test case
x = 8
n = 2
print(method())  # Expected output: True