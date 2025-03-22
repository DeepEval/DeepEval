def method(input_list):
    # Create a dictionary to count occurrences of each element
    occurrence_count = {}
    
    # Count occurrences of each element
    for num in input_list:
        if num in occurrence_count:
            occurrence_count[num] += 1
        else:
            occurrence_count[num] = 1
    
    # Filter out elements that occur more than once
    output = [num for num in input_list if occurrence_count[num] == 1]
    
    return output

# Test case for validation
input_list = [4, 5, 6, 7, 5, 6, 8, 9]
output = method(input_list)
print(output)  # Expected output: [4, 7, 8, 9]