def method(n, p):
    # Calculate 2^n % p using the built-in pow function for efficiency and handling large numbers
    output = pow(2, n, p)
    return output

# Example test case
def test_method():
    # Test case: 2^10 % 1000
    # Expected output: 24, because 2^10 = 1024 and 1024 % 1000 = 24
    n = 10
    p = 1000
    result = method(n, p)
    print(f"2^{n} % {p} = {result}")
    assert result == 24, "Test case failed!"
    
    # Add more test cases if necessary

# Run the test case
test_method()