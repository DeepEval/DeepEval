def method():
    def filter_list(input_list):
        # Filter the list to only include integers
        return [x for x in input_list if isinstance(x, int)]

    return filter_list

# Test case
def test_method():
    def test_filter_list():
        # assert method()([1, 'a', 2, 'b', 3]) == [1, 2, 3]
        # assert method()([1, 'a', 2.5, 'b', 3]) == [1, 2]
        # assert method()([1, 'a', 2, None, 3]) == [1, 2, 3]
        # print("All test cases pass")
        print(method()([1, 'a', 2, 'b', 3]))

    test_filter_list()

# Call the test function
test_method()