def method():
    # Example list of integers
    input_list = [2, 1, 3, 4, 2, 5, 3, 7, 8, 6]
    
    # Initialize the list to store rolling maximums
    rolling_max = []
    
    # Initialize the current maximum to a very low value
    current_max = float('-inf')
    
    # Iterate through the input list
    for num in input_list:
        # Update the current maximum
        if num > current_max:
            current_max = num
        # Append the current maximum to the rolling_max list
        rolling_max.append(current_max)
    
    # Return the final output
    return rolling_max

# Test case
output = method()
print("Input List:", [2, 1, 3, 4, 2, 5, 3, 7, 8, 6])
print("Rolling Maximum List:", output)