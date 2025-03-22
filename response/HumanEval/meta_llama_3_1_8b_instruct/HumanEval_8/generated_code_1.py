def method(lst):
    """
    This function calculates the sum and product of a given list of integers.
    
    Args:
    lst (list): A list of integers.
    
    Returns:
    tuple: A tuple consisting of the sum and product of all integers in the list.
    """
    
    # Check if the list is empty
    if not lst:
        # If the list is empty, return a tuple with sum as 0 and product as 1
        return 0, 1
    
    # Initialize sum and product variables
    total_sum = 0
    total_product = 1
    
    # Iterate over each number in the list
    for num in lst:
        # Add the number to the total sum
        total_sum += num
        # Multiply the number to the total product
        total_product *= num
    
    # Return a tuple with the total sum and total product
    return total_sum, total_product

# Test case
lst = [1, 2, 3, 4, 5]
output = method(lst)
print(output)  # Expected output: (15, 120)