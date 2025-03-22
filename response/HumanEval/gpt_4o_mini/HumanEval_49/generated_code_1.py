def method(n, p):
    # Calculate 2^n modulo p
    output = pow(2, n, p)
    return output

# Test case
n = 10  # Example value for n
p = 1000  # Example value for p
result = method(n, p)

print(f"2^{n} mod {p} = {result}")  # Expected output: 24 (since 1024 mod 1000 = 24)