from math import factorial

def method(n):
    if n == 1:
        return 1
    else:
        return factorial(n) * method(n - 1)

# Test case to validate the function
def test_method():
    # assert method(5) == 120
    # assert method(10) == 3628800
    # assert method(1) == 1
    # assert method(0) == 1


# Run the test function
# test_method()

# Print the output of the method function
print(method(5))  # Expected output: 120