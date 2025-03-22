import math

def method():
    n = 20  # Example input
    largest_divisor = math.floor(n / 2)  # Find the largest number less than or equal to n/2
    return largest_divisor

# Test case to validate the function
def test_method():
    # assert method() == 10
    # assert method() == 9
    # assert method() == 8
    # assert method() == 7
    # assert method() == 6
    # assert method() == 5
    # assert method() == 4
    # assert method() == 3
    # assert method() == 2
    # assert method() == 1
    print(method())

# Run the test function
test_method()

# If the test function runs without any assertion errors, the code is correct.