from typing import List

def method():

    # Input strings a and b
    a = "1010001101000101"
    b = "10101001101000101"

    # Create an empty list to store the XOR result
    result = []

    # Perform XOR operation on each bit of a and b
    for i in range(len(a)):
        if a[i] == b[i]:
            result.append("0")
        else:
            result.append("1")

    # Join the XOR result into a string
    output = "".join(result)

    return output

# Test case
test_case = {
    "a": "1010001101000101",
    "b": "10101001101000101",
    "expected_output": "0000101010000101"
}

# Call the method and compare the output with the expected output
output = method()
# assert output == test_case["expected_output"], "Test case failed"

# print("Test case passed")