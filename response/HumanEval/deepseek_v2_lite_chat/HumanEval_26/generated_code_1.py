def method(input_list):
    # Dictionary to store the frequency of each integer
    frequency = {}
    # List to store the final output
    output = []
    
    # Iterate through each element in the input list
    for num in input_list:
        # If the number is not in the frequency dictionary, add it with a count of 1
        if num not in frequency:
            frequency[num] = 1
            output.append(num)
        # If the number is already in the frequency dictionary, increment its count
        elif num in frequency:
            frequency[num] += 1
    
    return output

# Example usage:
output_list = method([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print(output_list)  # Output should be [1, 2, 3, 4]