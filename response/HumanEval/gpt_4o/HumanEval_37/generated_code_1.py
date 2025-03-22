def method(l):
    # Extract elements at even indices
    even_indices_elements = l[::2]
    # Sort the elements at even indices
    sorted_even_elements = sorted(even_indices_elements)

    # Initialize the output list
    output = []

    # Create a counter for the sorted even elements
    even_index_counter = 0

    # Iterate through the original list
    for i in range(len(l)):
        if i % 2 == 0:
            # Append sorted even element
            output.append(sorted_even_elements[even_index_counter])
            even_index_counter += 1
        else:
            # Append original odd index element
            output.append(l[i])

    return output

# Test case for validation
test_list = [10, 1, 3, 7, 2, 9, 8, 5, 6, 4]
expected_output = [2, 1, 3, 7, 6, 9, 8, 5, 10, 4]
output = method(test_list)
print("Output:", output)
print("Test Passed:", output == expected_output)