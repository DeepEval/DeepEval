import math

def method(a, b):
    # Calculate the greatest common divisor using math.gcd
    output = math.gcd(a, b)
    return output

# Test case
if __name__ == "__main__":
    # Example test case
    a = 48
    b = 18
    result = method(a, b)
    print(f"The GCD of {a} and {b} is: {result}")  # Expected output: 6