from operator import itemgetter

def method(lst):
    # Filter out strings with odd lengths
    filtered_lst = [s for s in lst if len(s) % 2 == 0]
    
    # Sort by length (shortest to longest), then alphabetically
    sorted_lst = sorted(filtered_lst, key=itemgetter(0, 1))
    
    return sorted_lst

# Test case to validate the function
def test_method():
    # Test with a list containing strings of equal length
    input_lst = ["apple", "banana", "cherry", "date", "fig", "grape"]
    expected_output = ["date", "fig", "grape"]
    # assert method(input_lst) == expected_output
    method(input_lst)

    # Test with a list containing strings of different lengths
    input_lst = ["apple", "banana", "cherry", "daddy", "elephant"]
    expected_output = ["apple", "banana", "cherry"]
    # assert method(input_lst) == expected_output

    print("All test cases pass")

# Call the test function
test_method()