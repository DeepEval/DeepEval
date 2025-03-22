def method():
    # List of elements
    elements = [4, 2, 9, 2, 5, 1, 8, 5, 9, 4]

    # Remove duplicates by converting the list to a set
    unique_elements = set(elements)

    # Convert the set back to a list
    sorted_elements = sorted(unique_elements)

    # Return the sorted list of unique elements
    return sorted_elements

# Test case
def test_method():
    result = method()
    # Expected output: [1, 2, 4, 5, 8, 9]
    assert result == [1, 2, 4, 5, 8, 9]
    print("Test passed!")

# Run the test
test_method()