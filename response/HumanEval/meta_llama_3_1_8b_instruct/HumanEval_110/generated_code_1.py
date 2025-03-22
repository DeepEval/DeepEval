def method(lst1, lst2):
    """
    This function checks whether it is possible to exchange elements between two lists to make the first list a list of only even numbers.

    Args:
        lst1 (list): The list that needs to be converted to all even numbers.
        lst2 (list): The list from which elements can be exchanged.

    Returns:
        str: "YES" if it is possible to exchange elements to make all elements in lst1 even, "NO" otherwise.
    """
    # Check if both inputs are lists
    if not isinstance(lst1, list) or not isinstance(lst2, list):
        raise TypeError("Both inputs must be lists")

    # Check if all elements in both lists are numbers
    if not all(isinstance(x, (int, float)) for x in lst1) or not all(isinstance(x, (int, float)) for x in lst2):
        raise TypeError("All elements in both lists must be numbers")

    # Separate even and odd numbers from lst1
    even_numbers_in_lst1 = [num for num in lst1 if num % 2 == 0]
    odd_numbers_in_lst1 = [num for num in lst1 if num % 2!= 0]

    # Check if it is possible to exchange elements to make all elements in lst1 even
    if len(odd_numbers_in_lst1) == 0 or len(even_numbers_in_lst1) >= len(odd_numbers_in_lst1):
        return "YES"
    else:
        # Count the number of odd numbers in lst2 that are greater than the odd numbers in lst1
        odd_numbers_in_lst2_greater_than_lst1 = sum(1 for num in lst2 if num % 2!= 0 and num > max(odd_numbers_in_lst1))
        
        # If there are enough odd numbers in lst2 that are greater than the odd numbers in lst1, return "YES"
        if odd_numbers_in_lst2_greater_than_lst1 >= len(odd_numbers_in_lst1):
            return "YES"
        else:
            return "NO"


# Test case
lst1 = [4, 6, 8, 10]
lst2 = [3, 5, 7, 9, 11, 12]
print(method(lst1, lst2))  # Output: "YES"

lst1 = [1, 3, 5, 7]
lst2 = [2, 4, 6, 8]
print(method(lst1, lst2))  # Output: "YES"

lst1 = [1, 3, 5, 7]
lst2 = [10, 11, 12]
print(method(lst1, lst2))  # Output: "NO"