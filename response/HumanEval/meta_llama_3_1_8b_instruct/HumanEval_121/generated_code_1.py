def method():
    """
    This function calculates the sum of all odd elements that are in even positions in a list of integers.

    Args:
    input_list (list): A non-empty list of integers.

    Returns:
    int: The sum of all odd elements that are in even positions.
    """
    
    # Initialize the sum variable to 0
    total_sum = 0
    
    # Iterate over the input list with both index and value
    for index, value in enumerate(input_list):
        
        # Check if the index is even and the value is odd
        if index % 2 == 0 and value % 2!= 0:
            
            # If the conditions are met, add the value to the total sum
            total_sum += value
    
    # Return the total sum
    return total_sum


# Test case for validation
input_list = [1, 2, 3, 4, 5, 6]
output = method()
print("Input List:", input_list)
print("Output (Sum of odd elements in even positions):", output)