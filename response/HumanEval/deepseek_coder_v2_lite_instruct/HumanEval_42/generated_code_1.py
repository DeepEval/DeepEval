def method():
    input_list = [1, 2, 3, 4, 5]  # Example input list
    output_list = [x + 1 for x in input_list]  # Increment each element by 1
    return output_list

# # Test case
# def test_method():
#     expected_output = [2, 3, 4, 5, 6]
#     assert method() == expected_output, f"Expected {expected_output}, but got {method()}"
#     print("Test passed!")

# # Run the test case
# test_method()

# Run the method
print(method())