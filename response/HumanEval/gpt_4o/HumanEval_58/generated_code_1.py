def method(list1, list2):
    # Use set intersection to find common elements
    common_elements = set(list1).intersection(set(list2))
    # Sort the resulting set of common elements and convert it to a list
    output = sorted(common_elements)
    return output

# Test case for validation
def test_method():
    list1 = [1, 2, 2, 3, 4, 5]
    list2 = [3, 4, 4, 5, 6, 7]
    expected_output = [3, 4, 5]
    assert method(list1, list2) == expected_output
    print("Test case passed!")

# Run the test case
test_method()