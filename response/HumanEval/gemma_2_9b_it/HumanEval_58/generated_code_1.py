def method():
    list1 = [1, 2, 3, 4, 5]
    list2 = [3, 5, 6, 7, 8]
    common_elements = set(list1).intersection(set(list2))
    sorted_common_elements = sorted(common_elements)
    return sorted_common_elements

# Test Case
output = method()
print(f"Sorted Unique Common Elements: {output}")