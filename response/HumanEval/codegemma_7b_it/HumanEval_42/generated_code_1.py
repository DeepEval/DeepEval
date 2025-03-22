# Import necessary packages
from typing import List

def method() -> List[int]:
    """Return list with elements incremented by 1."""
    input_list = [1, 2, 3, 4, 5]
    output_list = [x + 1 for x in input_list]
    return output_list

# Test case
test_input = [1, 2, 3, 4, 5]
test_output = method()
print(test_output)  # Output: [2, 3, 4, 5, 6]