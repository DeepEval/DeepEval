def method():
    # Sample list
    input_list = [4, 2, 2, 8, 3, 3, 1]
    
    # Get unique elements
    unique_elements = list(set(input_list))
    
    # Sort the unique elements
    sorted_unique_elements = sorted(unique_elements)
    
    # Return the sorted unique elements
    return sorted_unique_elements

# Test case
output = method()
print(output)  # Expected output: [1, 2, 3, 4, 8]