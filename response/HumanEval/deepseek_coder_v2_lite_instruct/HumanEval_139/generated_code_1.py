import math

def brazilian_factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= math.factorial(i)
    return result

def method():
    n = 5  # Example input
    output = brazilian_factorial(n)
    return output

# # Test case
# def test_brazilian_factorial():
#     assert brazilian_factorial(3) == 6 * 2 * 1  # 3! * 2! * 1! = 6 * 2 * 1 = 12
#     assert brazilian_factorial(4) == 24 * 6 * 2 * 1  # 4! * 3! * 2! * 1! = 24 * 6 * 2 * 1 = 288
#     assert brazilian_factorial(5) == 120 * 24 * 6 * 2 * 1  # 5! * 4! * 3! * 2! * 1! = 120 * 24 * 6 * 2 * 1 = 34560
#     print("All test cases passed!")

# # Running the test case
# test_brazilian_factorial()

print(method())  # Output should be 34560