def method(input_string):
    # Convert the string to a set to remove duplicates, then return the length of the set
    return len(set(input_string.lower()))

# Test case
def test_method():
    # assert method("Hello World!") == 9, "Test case 1 failed"
    # assert method("aA") == 3, "Test case 2 failed"
    # assert method("1234") == 4, "Test case 3 failed"
    # assert method("") == 0, "Test case 4 failed"
    # assert method("a") == 1, "Test case 5 failed"
    # print("All test cases pass")
    print(method("Hello World!"))
    

# Run the test case
test_method()