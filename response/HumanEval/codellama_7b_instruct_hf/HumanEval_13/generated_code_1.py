from math import gcd

def method(a, b):
    return gcd(a, b)

# Test case for validation
a = 12
b = 18
expected_output = 6

output = method(a, b)

if output == expected_output:
    print("Test case passed.")
else:
    print("Test case failed. Expected output:", expected_output, ", but got", output)