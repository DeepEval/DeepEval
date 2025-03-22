def method(input_list):
    # Convert the list to a set to get unique elements, then sort them
    unique_sorted_elements = sorted(set(input_list))
    return unique_sorted_elements

# Test case for validation
if __name__ == "__main__":
    test_input = [5, 3, 6, 2, 3, 1, 5, 6]
    output = method(test_input)
    print("Sorted unique elements:", output)