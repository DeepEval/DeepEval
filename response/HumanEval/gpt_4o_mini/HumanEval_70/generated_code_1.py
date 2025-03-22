def method(input_list):
    # Sort the input list
    sorted_list = sorted(input_list)
    output = []
    
    # Use two pointers
    left = 0
    right = len(sorted_list) - 1
    
    # Alternate between minimum and maximum
    while left <= right:
        if left == right:
            output.append(sorted_list[left])  # Append the last remaining element
        else:
            output.append(sorted_list[left])  # Append minimum
            output.append(sorted_list[right])  # Append maximum
        left += 1
        right -= 1
    
    return output

# Test case
test_input = [3, 1, 4, 1, 5, 9, 2, 6, 5]
output = method(test_input)
print(output)  # Example output: [1, 9, 1, 6, 2, 5, 3, 4]