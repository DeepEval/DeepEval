def method(numbers):
    # Check if the list is sorted in ascending order
    is_sorted = all(numbers[i] <= numbers[i + 1] for i in range(len(numbers) - 1))
    
    # Check for duplicates by converting the list to a set and comparing lengths
    has_no_multiple_duplicates = len(set(numbers)) == len(numbers)
    
    # Both conditions must be true
    output = is_sorted and has_no_multiple_duplicates
    return output

# Test case for validation
test_case = [1, 2, 2, 3, 4, 5]
print(method(test_case))  # Expected output: False, because '2' appears more than once