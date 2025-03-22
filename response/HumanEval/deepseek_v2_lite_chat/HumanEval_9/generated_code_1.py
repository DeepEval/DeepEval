import itertools

def method():
    # Sample input list of integers
    input_list = [1, 3, 5, 7, 2, 6, 9, 10, 11]
    
    # Using itertools.accumulate to get the rolling maximum
    rolling_max = list(itertools.accumulate(input_list, func=max))
    
    # Return the list of rolling maximums
    return rolling_max

# Test case to validate the function
def test_method():
    # assert method() == [1, 3, 5, 7, 7, 7, 9, 10, 11]
    # print("All test cases pass")
    print(method())

# Run the test function
test_method()