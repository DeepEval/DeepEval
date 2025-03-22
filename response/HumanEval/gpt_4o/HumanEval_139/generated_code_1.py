import math

def method(n):
    brazilian_factorial = 1
    for i in range(1, n + 1):
        brazilian_factorial *= math.factorial(i)
    
    return brazilian_factorial

# Test case for validation
def test_brazilian_factorial():
    # For n = 3, Brazilian factorial should be 3! * 2! * 1! = 6 * 2 * 1 = 12
    assert method(3) == 12, "Test case 1 failed"
    # For n = 4, Brazilian factorial should be 4! * 3! * 2! * 1! = 24 * 6 * 2 * 1 = 288
    assert method(4) == 288, "Test case 2 failed"
    # For n = 5, Brazilian factorial should be 5! * 4! * 3! * 2! * 1! = 120 * 24 * 6 * 2 * 1 = 34560
    assert method(5) == 34560, "Test case 3 failed"
    print("All test cases passed.")

# Run test case
test_brazilian_factorial()