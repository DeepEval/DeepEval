def method(lst):
    # Initialize sum and product
    total_sum = 0
    total_product = 1
    
    # Iterate through the list to calculate sum and product
    for num in lst:
        total_sum += num
        total_product *= num
    
    # Return the result as a tuple
    return (total_sum, total_product)

# Test case for validation
# Example list
test_list = [1, 2, 3, 4]

# The expected output should be (10, 24) because:
# Sum: 1 + 2 + 3 + 4 = 10
# Product: 1 * 2 * 3 * 4 = 24

# Run the method function with the test case
output = method(test_list)
print(output)  # Output should be (10, 24)

# Additional test case with an empty list
empty_list = []
empty_output = method(empty_list)
print(empty_output)  # Output should be (0, 1)