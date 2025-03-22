import math

def method(n, p):

    # Calculate 2^n modulo p using the bitwise right shift operator
    output = pow(2, n, p)

    return output


# Test case
n = 3
p = 1000000007
print(method(n, p))