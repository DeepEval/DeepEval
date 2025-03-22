def method():
    input_list = [1, -2, 3, -4, 5, -6]
    output = [num for num in input_list if num > 0]
    return output

# Test case
def test_method():
    result = method()
    # assert result == [1, 3, 5], f"Expected [1, 3, 5], but got {result}"
    # print("Test passed!")

# Running the test case
test_method()