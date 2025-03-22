def method():
    """
    Sorts the given list of integers in ascending order according to the sum of their digits.
    If there are several items with similar sum of their digits, order them based on their index in original list.

    Returns:
        list: The sorted list of integers.
    """
    
    # Import necessary packages
    import operator
    
    # Define the original list of integers
    original_list = [12, 3, 7, 15, 22, 21, 6, 9]
    
    # Calculate the sum of digits for each number in the list
    list_with_digit_sum = [(num, sum(int(digit) for digit in str(num)), i) for i, num in enumerate(original_list)]
    
    # Sort the list based on the sum of digits and index
    sorted_list = sorted(list_with_digit_sum, key=operator.itemgetter(1, 2))
    
    # Extract the sorted numbers from the list
    sorted_numbers = [num for num, _, _ in sorted_list]
    
    return sorted_numbers


# Test case
print(method())