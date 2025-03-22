def method(lst):
    # Initialize the sum to 0
    even_sum = 0
    
    # Iterate over the list using indices
    for i in range(len(lst)):
        # Check if the index is odd
        if i % 2 != 0:
            # Check if the element at that index is even
            if lst[i] % 2 == 0:
                # Add the element to the sum
                even_sum += lst[i]
                
    # Return the final sum
    return even_sum

# Test case for validation
def test_method():
    # Example test case
    lst = [1, 4, 3, 6, 5, 8, 7, 10]
    # Odd indices are 1 (4), 3 (6), 5 (8), 7 (10)
    # Even numbers at these indices are 4, 6, 8, 10
    # Sum is 4 + 6 + 8 + 10 = 28
    assert method(lst) == 28
    print("Test case passed!")

# Run the test case
test_method()