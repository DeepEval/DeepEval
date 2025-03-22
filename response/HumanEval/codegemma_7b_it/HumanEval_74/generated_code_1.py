from collections import Counter

def method(list1, list2):
    """
    This function accepts two lists of strings and returns the list with total number of chars in the all strings of the list less than the other list. 
    if the two lists have the same number of chars, return the first list.

    Args:
        list1: The first list of strings.
        list2: The second list of strings.

    Returns:
        The list with total number of chars in the all strings of the list less than the other list. If the two lists have the same number of chars, return the first list.
    """

    total_chars_list1 = sum(len(word) for word in list1)
    total_chars_list2 = sum(len(word) for word in list2)

    if total_chars_list1 < total_chars_list2:
        output = list1
    elif total_chars_list2 < total_chars_list1:
        output = list2
    else:
        output = list1

    return output


# Test case for validation
list1 = ["apple", "banana", "cherry"]
list2 = ["dog", "cat", "fish"]
print(method(list1, list2))