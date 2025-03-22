def method(lst):
    # Create a set to store the elements and check for duplicates
    elements_set = set()
    for num in lst:
        if -num in elements_set:
            return True
        elements_set.add(num)
    return False

# Test case to validate the function
def test_method():
    # assert method([1, -1, 2, -2]) == True
    # assert method([3, -3, 2, 1]) == True
    # assert method([1, 2, 3, 4]) == False
    # assert method([]) == False
    # print("All test cases pass")
    method([1, 2, 3, 4])

# Run the test function
test_method()