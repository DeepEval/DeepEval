def method():
    """
    Sorts an array of non-negative integers according to the number of ones in their binary representation in ascending order.
    For similar number of ones, sorts based on decimal value.
    
    Args:
        None
    
    Returns:
        list: The sorted array of integers.
    """
    
    # Import necessary packages
    import math
    
    # Define the function to count the number of ones in binary representation
    def count_ones(n):
        return bin(n).count('1')
    
    # Define the sorting key as a tuple
    def sorting_key(n):
        return (count_ones(n), n)
    
    # Example array to be sorted
    array = [10, 9, 5, 6, 7, 12]
    
    # Sort the array using the custom sorting key
    sorted_array = sorted(array, key=sorting_key)
    
    # Return the sorted array
    return sorted_array

# Test case
output = method()
print(output)