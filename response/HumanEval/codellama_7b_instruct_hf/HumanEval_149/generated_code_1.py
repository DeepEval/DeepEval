import functools

def method(str_list):
    # Filter out strings with odd lengths
    filtered_list = [word for word in str_list if len(word) % 2 == 0]

    # Sort the filtered list by word length (ascending) and alphabetically if lengths are equal
    sorted_list = sorted(filtered_list, key=functools.cmp_to_key(lambda x, y: (len(x) - len(y)) or (x < y)))

    return sorted_list

# Test case
str_list = ["apple", "banana", "orange", "kiwi", "grapefruit"]
sorted_list = method(str_list)
print(sorted_list)