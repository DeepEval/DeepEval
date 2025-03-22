def method(list1, list2):
    # Calculate the total number of characters in each list
    total_chars_list1 = sum(len(s) for s in list1)
    total_chars_list2 = sum(len(s) for s in list2)

    # Compare the total characters and return the appropriate list
    if total_chars_list1 < total_chars_list2:
        return list1
    elif total_chars_list1 > total_chars_list2:
        return list2
    else:
        return list1  # If both totals are equal, return the first list

# Test case
list_a = ["hello", "world"]
list_b = ["python", "is", "great"]

output = method(list_a, list_b)
print(output)  # Expected output: ['hello', 'world'] since 10 < 15