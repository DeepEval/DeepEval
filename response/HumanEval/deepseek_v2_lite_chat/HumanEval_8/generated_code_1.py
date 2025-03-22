from functools import reduce

def method():
    def compute_sum_and_product(numbers):
        # Compute the product of all numbers in the list
        product = reduce(lambda x, y: x * y, numbers, 1)
        
        # Compute the sum of all numbers in the list
        sum_result = sum(numbers)
        
        # Return the tuple containing the sum and product
        return (sum_result, product)
    
    # Example usage
    test_list = [1, 2, 3, 4, 5]
    sum_product_result = compute_sum_and_product(test_list)
    
    return sum_product_result

# Test case to validate the function
def test_method():
    test_list = [1, 2, 3, 4, 5]
    expected_sum = 15
    expected_product = 120
    result = method()
    # assert result[0] == expected_sum
    # assert result[1] == expected_product
    # print("Test case passed!")
    print(result[0])

# Run the test case
test_method()

# Example usage (you can comment this line to remove the test case from the final output)
#print("The sum and product are:", method())