def method(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b

# Test case for validation
def test_method():
    # Test for the 10th Fibonacci number, which should be 55
    assert method(10) == 55, "Test case failed!"
    print("Test case passed!")

# Run the test case
test_method()