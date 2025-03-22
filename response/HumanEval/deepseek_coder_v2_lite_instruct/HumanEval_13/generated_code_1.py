def method():
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Example usage:
    a = 56
    b = 98
    output = gcd(a, b)
    return output

# Test case
print(method())  # Expected output: 14