import math

def method(n):
    if n <= 0:
        raise ValueError("n must be greater than 0")
    
    brazilian_fact = 1
    for i in range(1, n + 1):
        brazilian_fact *= math.factorial(i)
    
    return brazilian_fact

# Test case for validation
if __name__ == "__main__":
    # Test with n = 3
    n = 3
    output = method(n)
    print(f"The Brazilian factorial for n={n} is: {output}")  # Expected output: 12 (1! * 2! * 3!)