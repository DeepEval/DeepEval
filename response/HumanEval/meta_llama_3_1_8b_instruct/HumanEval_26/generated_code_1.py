def method():
    """
    Removes all elements that occur more than once from a list of integers
    and keeps the order of elements left the same as in the input.
    
    Args: None
    
    Returns: A list of integers where each integer occurs only once.
    """
    
    # Initialize an empty dictionary to store the count of each element
    count_dict = {}
    
    # Initialize an empty list to store the output
    output = []
    
    # Iterate over the input list
    for num in [1, 2, 2, 3, 4, 4, 5, 6, 6, 7]: 
        # Count the occurrence of each element in the dictionary
        if num in count_dict:
            count_dict[num] += 1
        else:
            count_dict[num] = 1
    
    # Iterate over the input list again
    for num in [1, 2, 2, 3, 4, 4, 5, 6, 6, 7]: 
        # Check if the count of the element is 1
        if count_dict[num] == 1:
            # Add the element to the output list
            output.append(num)
            # Decrement the count of the element in the dictionary
            count_dict[num] -= 1
    
    return output

# Test the function
print(method())  # Output: [1, 3, 5, 7]