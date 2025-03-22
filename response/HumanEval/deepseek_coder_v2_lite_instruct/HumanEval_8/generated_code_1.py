def method():
    # Import necessary packages
    # No external packages are needed for this task
    
    # Define the input list of integers
    numbers = [1, 2, 3, 4]  # Example list of integers
    
    # Initialize sum and product
    total_sum = 0
    total_product = 1
    
    # Calculate the sum and product
    for num in numbers:
        total_sum += num
        total_product *= num
    
    # Return the result as a tuple
    output = (total_sum, total_product)
    return output

# # Test case to validate the function
# def test_method():
#     # Test case 1: Normal case with positive numbers
#     assert method() == (10, 24), "Test case 1 failed"
    
#     # Test case 2: Edge case with an empty list
#     empty_list = []
#     assert method(empty_list) == (0, 1), "Test case 2 failed"
    
#     # Test case 3: Edge case with a list containing zero
#     zero_list = [0, 1, 2, 3]
#     assert method(zero_list) == (6, 0), "Test case 3 failed"
    
#     print("All test cases passed!")

# # Run the test case
# test_method()

method()