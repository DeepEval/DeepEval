def method(input_list):
    # Filter out strings with odd lengths
    filtered_list = [s for s in input_list if len(s) % 2 == 0]
    
    # Sort the list by length and then alphabetically
    sorted_list = sorted(filtered_list, key=lambda x: (len(x), x))
    
    return sorted_list

# Test case
input_list = ["apple", "banana", "cherry", "date", "fig", "grape"]
output = method(input_list)
print(output)  # Expected output: ['banana', 'cherry', 'date', 'grape']