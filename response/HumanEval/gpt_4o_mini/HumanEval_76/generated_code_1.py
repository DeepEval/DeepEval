def method(n, x):
    if n <= 0 or x < 0:
        return False
    if n == 1:
        return x == 1  # 1^k = 1 for any k >= 0
    
    k = 0
    power = 1  # n^0 = 1
    while power < x:
        k += 1
        power = n ** k
        
    return power == x

# Test case for validation
n = 3
x = 9
output = method(n, x)
print(f"Is {x} a simple power of {n}? {output}")  # Expected output: True (since 3^2 = 9)