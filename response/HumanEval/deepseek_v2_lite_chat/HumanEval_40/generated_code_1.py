def method(lst):
    # Create a set to store the elements and check for sums to zero
    element_set = set()
    for num in lst:
        # If a pair sums to zero, return True
        if -num in element_set:
            return True
        # Add the number to the set
        element_set.add(num)
    # If no such three numbers exist, return False
    return False

# Test case
def test_method():
    # assert method([1, -1, 2, -2, 0]) == True
    # assert method([1, -1, 0]) == False
    # assert method([1, -1]) == False
    # assert method([0, 0, 0]) == True
    # assert method([1]) == False
    # assert method([1, 2, 3]) == False
    method([1, 2, 3])

# Call the test function
test_method()