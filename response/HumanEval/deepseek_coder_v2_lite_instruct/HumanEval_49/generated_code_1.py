def method(n, p):
    # Calculate 2^n modulo p
    result = pow(2, n, p)
    return result

# Test case
n = 10
p = 1000
output = method(n, p)
print(f"2^{n} mod {p} = {output}")