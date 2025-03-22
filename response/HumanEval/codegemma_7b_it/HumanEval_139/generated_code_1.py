import math

def method():
    # Get the input number from the user
    # n = int(input("Enter a non-negative integer: "))
    n = 5

    # Validate the input
    if n < 0:
        raise ValueError("Input number must be non-negative")

    # Calculate the Brazilian factorial
    brazilian_factorial = math.factorial(n)
    for i in range(n-1, 0, -1):
        brazilian_factorial *= math.factorial(i)

    # Return the output
    return brazilian_factorial

# Test case
test_case = 5
expected_output = 14400
actual_output = method()

print("Test case:", test_case)
print("Expected output:", expected_output)
print("Actual output:", actual_output)

if expected_output == actual_output:
    print("Test case passed")
else:
    print("Test case failed")