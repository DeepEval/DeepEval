from collections import Counter

def method(lst):
    """
    This function finds the greatest integer in a given list that has a frequency greater than or equal to its value.
    
    Parameters:
    lst (list): A non-empty list of positive integers.
    
    Returns:
    int: The greatest integer that meets the condition. If no such value exists, returns -1.
    """
    
    # Count the frequency of each integer in the list
    freq = Counter(lst)
    
    # Initialize the result as -1
    result = -1
    
    # Iterate over the list in reverse order to find the greatest integer
    for num in sorted(lst, reverse=True):
        # Check if the frequency of the current integer is greater than or equal to its value
        if freq[num] >= num:
            # Update the result if the current integer meets the condition
            result = num
            # Break the loop since we found the greatest integer that meets the condition
            break
    
    return result

# Test case
lst = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
print(method(lst))  # Output: 4