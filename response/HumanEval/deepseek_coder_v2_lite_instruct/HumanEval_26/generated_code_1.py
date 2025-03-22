def method():
    input_list = [1, 2, 2, 3, 4, 4, 5]
    element_count = {}
    
    # Count the occurrences of each element
    for num in input_list:
        element_count[num] = element_count.get(num, 0) + 1
    
    # Filter out elements that occur more than once
    output_list = [num for num in input_list if element_count[num] == 1]
    
    return output_list

# Test case
output = method()
print("Output:", output)  # Expected output: [1, 3, 5]