from itertools import groupby

def method():
    numbers = [1, 2, 3, 4, 5]
    delimeter = "#"
    output = list(map("".join, groupby(numbers, key=lambda x, y: y - x)))
    output = [delimeter.join(list(group)) for _, group in groupby(output, key=len)]
    return output

# Test Case
input_list = [1, 2, 3, 4, 5]
delimeter = "#"
expected_output = ['1#2', '3#4', '5']

actual_output = method()

if actual_output == expected_output:
    print("Test Case Passed")
else:
    print("Test Case Failed")