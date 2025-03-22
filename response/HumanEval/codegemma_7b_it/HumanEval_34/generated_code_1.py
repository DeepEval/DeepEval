from collections import OrderedDict

def method():
    input_list = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    sorted_unique_elements = sorted(set(input_list))
    return sorted_unique_elements

# Test case
# assert method() == [1, 2, 3, 4, 5]