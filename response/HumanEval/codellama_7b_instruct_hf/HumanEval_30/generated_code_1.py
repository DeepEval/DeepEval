import numpy as np

def method():
    # Create a list of positive numbers
    input_list = [1, 2, -3, 4, -5, 6, -7, 8]

    # Use boolean indexing to filter out negative numbers
    positive_list = [number for number in input_list if number > 0]

    # Return the positive numbers
    return positive_list

# Test case
test_case = method()
print(test_case)  # Output: [1, 2, 4, 6, 8]