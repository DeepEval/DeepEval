def method():
    list1 = ["apple", "banana", "cherry"]
    list2 = ["dog", "cat", "bird", "fish"]
    output = find_shorter_list(list1, list2)
    return output

def find_shorter_list(list1, list2):
    """
    Compares the total number of characters in two lists of strings.

    Args:
        list1: The first list of strings.
        list2: The second list of strings.

    Returns:
        The list with the smaller total number of characters, or the first list if they are equal.
    """
    char_count1 = sum(len(word) for word in list1)
    char_count2 = sum(len(word) for word in list2)

    if char_count1 < char_count2:
        return list1
    elif char_count2 < char_count1:
        return list2
    else:
        return list1

# Test Case
print(method())